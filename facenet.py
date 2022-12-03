"""
Facial recognition with FaceNet in Keras, TensorFlow, or TensorRT.
"""

from __future__ import annotations

import ntpath
import os
import threading
from timeit import default_timer as timer

import cv2
import numpy as np
from termcolor import colored
from loguru import logger
from tqdm import tqdm

import util

try:
    import pycuda.autoinit  # noqa
    import pycuda.driver as cuda  # noqa
except (ModuleNotFoundError, ImportError) as e:
    logger.debug("[GPU] pycuda autoinit failed")

try:
    import tensorrt as trt  # noqa
except (ModuleNotFoundError, ImportError) as e:
    logger.debug("[GPU] tensorrt import failed")


class FaceNet:
    """
    Facial recognition. Most work (facial detection, graphics, etc) is handled
    by `util`. CNN embedding by FaceNet is handled here.
    """

    @util.data.print_time("Model load time")
    def __init__(self,
                 model_path: str,
                 data: util.data.Database,
                 input_name: str = None,
                 output_name: str = None,
                 input_shape: tuple = None,
                 gpu_alloc: bool = False):
        """
        Initializes FaceNet object.

        :param model_path: path to file (.pb, .h5, .tflite, or .engine)
        :param data: Database object that contains facial recognition database.
            Can use empty database but must load data manually
        :param input_name: name of input tensor, ignore if not using TF mode
            (default: None)
        :param output_name: name of output tensor, ignore if not using TF mode
            (default: None)
        :param input_shape: shape of input tensor, ignore if not using TF mode
            (default: None)
        :param gpu_alloc: if true, allows memory growth on GPU in TF mode
            (default: False)
        """
        if gpu_alloc:
            import tensorflow as tf  # noqa

            try:
                gpus = tf.config.experimental.list_physical_devices("GPU")
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as err:
                print(err)

        if ".h5" in model_path:
            self._keras_init(model_path)
        elif ".tflite" in model_path:
            self._tflite_init(model_path)
        elif ".pb" in model_path:
            self._tf_init(model_path, input_name, output_name, input_shape)
        elif ".engine" in model_path:
            self._trt_init(model_path, input_shape)
        else:
            raise TypeError("model must be an .h5, .pb, or .engine file")

        logger.debug(f"Inference backend is {self.mode}")

        self.db = data

    @property
    def data(self) -> dict:
        """
        Returns database data.

        :return: data in dict format
        """
        return self.db.data

    def _keras_init(self, filepath: str):
        """
        Initializes model from keras engine.
        """
        import tensorflow.compat.v1 as tf  # noqa

        self.mode = "keras"
        self.facenet = tf.keras.models.load_model(filepath)
        self.img_shape = self.facenet.input_shape[1:3]

    def _tflite_init(self, filepath: str):
        """
        Initializes model from tflite engine.
        """
        import tensorflow.compat.v1 as tf  # noqa

        self.mode = "tflite"
        self.facenet = tf.lite.Interpreter(model_path=filepath)
        self.facenet.allocate_tensors()

        self.input_details = self.facenet.get_input_details()
        self.output_details = self.facenet.get_output_details()
        self.img_shape = self.input_details[0]["shape"].tolist()[1:-1]

    def _tf_init(self,
                 filepath: str,
                 input_name: str = "input",
                 output_name: str = "embeddings",
                 input_shape: tuple = (160, 160)):
        """
        Initializes model from pb graphdef (TF v1).
        """
        import tensorflow.compat.v1 as tf  # noqa

        self.mode = "tf"

        self.input_name = input_name
        self.output_name = output_name
        self.img_shape = input_shape

        with tf.gfile.FastGFile(filepath, "rb") as graph_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(graph_file.read())

        self.sess = tf.keras.backend.get_session()

        tf.import_graph_def(graph_def, name="")
        self.facenet = self.sess.graph

    def _trt_init(self, filepath, input_shape):
        """
        Initializes model from tensorrt engine.
        """
        self.mode = "trt"
        try:
            self.dev_ctx = cuda.Device(0).make_context()
            self.stream = cuda.Stream()
            trt_logger = trt.Logger(trt.Logger.INFO)
            runtime = trt.Runtime(trt_logger)

            with open(filepath, "rb") as model:
                self.facenet = runtime.deserialize_cuda_engine(model.read())

            self.context = self.facenet.create_execution_context()

            self.h_input = cuda.pagelocked_empty(
                trt.volume(self.context.get_binding_shape(0)), dtype=np.float32
            )
            self.h_output = cuda.pagelocked_empty(
                trt.volume(self.context.get_binding_shape(1)), dtype=np.float32
            )

            self.d_input = cuda.mem_alloc(self.h_input.nbytes)
            self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        except NameError:
            raise ValueError("trt mode requested but not available")

        self.img_shape = input_shape

    def embed(self, imgs: np.ndarray) -> np.ndarray:
        """
        Embeds cropped face.

        :param imgs: list of cropped faces with shape (b, h, w, 3)
        :returns: embedding as array with shape (1, -1)
        """

        if self.mode == "keras":
            embeds = self.facenet.predict(imgs, batch_size=len(imgs))

        elif self.mode == "tf":
            out = self.facenet.get_tensor_by_name(self.output_name)
            embeds = self.sess.run(out, feed_dict={self.input_name: imgs})

        elif self.mode == "tflite":
            imgs = imgs.astype(np.float32)
            self.facenet.set_tensor(self.input_details[0]["index"], imgs)
            self.facenet.invoke()
            embeds = self.facenet.get_tensor(self.output_details[0]["index"])

        else:
            if len(imgs) != 1:
                raise NotImplementedError("trt batch not yet supported")
            threading.Thread.__init__(self)
            self.dev_ctx.push()

            np.copyto(self.h_input, imgs.astype(np.float32).ravel())
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            self.context.execute_async(
                batch_size=1,
                bindings=[int(self.d_input), int(self.d_output)],
                stream_handle=self.stream.handle,
            )
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()
            self.dev_ctx.pop()

            embeds = np.copy(self.h_output)

        return embeds.reshape(len(imgs), -1)

    def predict(self,
                img: np.ndarray,
                detector: util.detection.FaceDetector,
                margin: int = 10,
                flip: bool = False,
                verbose: bool = True) -> tuple[np.ndarray | None, dict | None]:
        """
        Embeds and normalizes an image from path or array.

        :param img: image to be predicted on (BGR image without batch dim)
        :param detector: FaceDetector object
        :param margin: margin for MTCNN face cropping (default: 10)
        :param flip: flip and concatenate or not (default: False)
        :param verbose: verbosity (default: True)
        :returns: normalized embeddings, facial coordinates
        """

        cropped_faces, face_coords = detector.crop_face(img, margin, flip, verbose)
        if cropped_faces is None:
            return None, None

        start = timer()

        normalized = self.db.normalize(np.array(cropped_faces))
        embeds = self.embed(normalized)
        embeds = self.db.dist_metric.apply_norms(embeds, batch=True)

        if verbose:
            elapsed = round(1000.0 * (timer() - start), 2)
            time = colored(f"{elapsed} ms", attrs=["bold"])
            vecs = f"{len(embeds)} vector{'s' if len(embeds) > 1 else ''}"
            print(f"Embedding time ({vecs}): {time}")

        return embeds, face_coords

    def recognize(self,
                  img: np.ndarray,
                  **kwargs) -> tuple:
        """
        Facial recognition on input image.

        :param img: image array in BGR mode (without batch dimension)
        :param kwargs: will be passed to self.predict
        :returns: face, is recognized, best match, time elapsed
        """
        start = timer()

        is_recognized = None
        best_match = None
        face = None

        try:
            embeds, face = self.predict(img, **kwargs)
            if embeds is not None:
                nearest, best_match = self.db.nearest_embed(embeds)
                dists = self.db.dist_metric.distance(embeds, nearest, batch=True)
                dist = np.average(dists)
                is_recognized = dist <= self.db.metadata["alpha"]

                if kwargs.get("verbose", True) and dist:
                    info = colored(
                        f"{round(dist, 4)} ({best_match}), "
                        f"α={self.db.metadata['alpha']}",
                        color="green" if is_recognized else "red",
                    )
                    print(f"{self.db.dist_metric}: {info}")

        except (ValueError, cv2.error) as error:
            incompatible = "query data dimension"
            if isinstance(error, ValueError) and incompatible in str(error):
                raise ValueError("Current model incompatible with database")
            elif isinstance(error, cv2.error) and "resize" in str(error):
                print("Frame capture failed")
            else:
                raise error

        elapsed = round(1000.0 * (timer() - start), 4)
        return face, is_recognized, best_match, elapsed

    def run_on_stream(self,
                      cap: object,
                      resize: float = 1.0,
                      flip: bool = False,
                      use_graphics: bool = True,
                      detector: util.detection.FaceDetector = None,
                      drawer: util.visuals.GraphicsRenderer = None,
                      logger: util.log.Logger = None) -> util.log.Logger:
        """
        Runs facial recognition on live-streamed data. Spawns output frame
        video and returns logging history.

        :param cap: cv2.VideoCapture object or util.visual.Camera object, 
            must support `read` and `release` methods
        :param resize: resizes window by scale factor, used for faster
            inference (default: 1.0)
        :param flip: if True, flips each frame (default: False)
        :param use_graphics: if True, display output video stream (default: True)
        :param detector: if specified, uses this FaceDetector object 
            to detect faces. If None, initializes new FaceDetector with
            default parameters (default: None).
        :param drawer: if specified, uses this GraphicsRenderer object 
            to draw box on output frame. If None, initializes new GraphicsRenderer with
            default parameters (default: None).
        :param drawer: if specified, uses this Logger object to record activity. 
            If None, initializes new Logger with default parameters (default: None).
        :return: Logger object with detection history
        """
        assert self.db.data, "data must be provided"
        assert 0.0 <= resize <= 1.0, "resize must be in [0., 1.]"

        if detector is None:
            detector = util.detection.FaceDetector(img_shape=self.img_shape)
        if drawer is None:
            drawer = util.visuals.GraphicsRenderer(resize=resize)
        if logger is None:
            logger = util.log.Logger()

        while True:
            _, frame = cap.read()
            cframe = frame.copy()

            # resize frame
            if resize != 1:
                frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)

            # facial detection and recognition
            info = self.recognize(frame, detector=detector, flip=flip)
            face, is_recognized, best_match, elapsed = info

            # logging and pbar
            if is_recognized and util.detection.is_looking(face):
                logger.log(best_match)

            # graphics
            if use_graphics:
                drawer.add_graphics(cframe, *info)
                cv2.imshow("FaceNet", cframe)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()

        return logger

    def embed_dir(self,
                  img_dir: str,
                  people: list[str] = None,
                  return_fails: bool = False,
                  **kwargs) -> tuple[dict, list] | dict:
        """
        Generates facial embedding vectors from directory of images. Faces are 
        detected by MTCNN and do not need to be pre-cropped.

        :param img_dir: directory of images. In the return dict, each key will
        correspond to an image file name with the extension removed. All images
        must be in .png or .jpg format.
        :param people: list of names to include in the embedding process. Each 
        entry must exactly match the image filenames. If None, loops over 
        all images in the directory (default: None).
        :param return_fails: if True, returns names of files for which detection
        or recognition failed (default: False).
        :param kwargs: extra params to `self.predict`
        :return: dict with keys as filenames (stripped of extensions) mapped to
        embedding vectors. If `return_fails`, also returns list of failed files.
        """
        if people is None:
            people = [
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if f.endswith("jpg") or f.endswith("png")
            ]
        if "detector" not in kwargs:
            detector = util.detection.FaceDetector(img_shape=self.img_shape)
            kwargs["detector"] = detector
        if "verbose" not in kwargs:
            kwargs["verbose"] = False

        data = {}
        no_faces = []

        for person in tqdm(people):
            if not person.endswith("jpg") and not person.endswith("png"):
                logger.error(f"'{person}' not a jpg or png image")
                no_faces.append(person)
            elif os.path.getsize(person) > 1e8:
                logger.error(f"'{person}' too large (> 100M bytes)")
                no_faces.append(person)
            else:
                embeds, _ = self.predict(cv2.imread(person), **kwargs)
                if embeds is not None:
                    person = ntpath.basename(person)
                    person = person.replace(".jpg", "").replace(".png", "")
                    data[person] = embeds.reshape(len(embeds), -1)
                else:
                    no_faces.append(person)
                    logger.error(f"no face detected for '{person}'")

        if return_fails:
            return data, no_faces
        return data
