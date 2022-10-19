"""
Facial recognition database management.
"""

from __future__ import annotations

import functools
from timeit import default_timer as timer
import json
from typing import Callable

from loguru import logger
import numpy as np
from sklearn import svm, neighbors

from util import distance


def strip_id(name: str,
             split: str = "-",
             return_index: bool = False) -> tuple | str:
    try:
        idx = name.rfind(split)
        if idx == -1:
            raise ValueError
        id_ = int(name[idx + 1:])
        name = name[:idx]
    except ValueError:
        id_ = None

    if return_index:
        return name, id_
    return name


def print_time(message: str) -> Callable:
    def _timer(func):
        @functools.wraps(func)
        def _func(*args, **kwargs):
            start = timer()
            result = func(*args, **kwargs)
            logger.debug(f"{message}: {round(timer() - start, 4)}s")
            return result

        return _func

    return _timer


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj: object) -> object:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


class Database:
    DEFAULT_METADATA = {
        "metric": "cosine",
        "normalize": True,
        "alpha": 0.5
    }

    def __init__(self, classifier_type: str = "svm"):
        self._db = {}
        self.metadata = {}
        self.classifier_type = classifier_type
        self.classifier = None
        self.dist_metric = None

        # Set up metadata in case data isn't provided later
        self.set_data(data={}, train_classifier=False, calc_mean=False)

    @property
    def data(self) -> dict:
        return self._db

    @property
    def names(self) -> list[str]:
        return list(self._db.keys())

    @property
    def embeds(self) -> list[np.ndarray]:
        return list(self._db.values())

    def __getitem__(self, item: str) -> np.ndarray:
        return self._db[item]

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        self.set_data(data["data"], data["metadata"])
        logger.info(f"Loaded data from '{path}'")

    def set_data(self,
                 data: dict[str, list[list]],
                 metadata: dict = None,
                 calc_mean: bool = True,
                 train_classifier: bool = True):

        for person, embeds in data.items():
            self.add(person, embeds, train_classifier=False, concat=True)

        if train_classifier:
            self._train_classifier()

        self.metadata = metadata or {}
        self.metadata = {**Database.DEFAULT_METADATA, **self.metadata}

        if calc_mean:
            embeds = list(self.all_pairs().values())
            self.metadata["mean"] = np.average(embeds)

        self.dist_metric = distance.DistMetric(
            metric=self.metadata["metric"],
            normalize=self.metadata["normalize"],
            mean=self.metadata.get("mean")
        )

    def dump(self, path: str):
        with open(path, mode="w+", encoding="utf-8") as f:
            data = {"metadata": self.metadata, "data": self.data}
            json.dump(data, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
        logger.info(f"Wrote data to '{path}'")

    def all_pairs(self) -> dict:
        all_pairs = {}
        for name, embeds in self.data.items():
            for i, embed in enumerate(embeds):
                name_ext = f"{name}-{i}"
                all_pairs[name_ext] = embed
        return all_pairs

    def add(self,
            name: str,
            embeddings: list[np.ndarray | list],
            concat: bool = False,
            train_classifier: bool = False):
        # Name is in the form "last_first" or "last_first-index"
        name = strip_id(name)

        embeds = np.array(embeddings).reshape(len(embeddings), -1)
        if concat:
            try:
                self._db[name] = np.concatenate([self._db[name], embeds])
            except KeyError:
                self._db[name] = embeds
        else:
            self._db[name] = embeds

        if train_classifier:
            self._train_classifier()

    def remove(self,
               name: str,
               train_classifier: bool = True):
        del self._db[name]

        if train_classifier:
            self._train_classifier()

    def nearest_embed(self, embeds: np.ndarray) -> tuple[np.ndarray, str]:
        best_match = self.classifier.predict(embeds)[0]
        name, i = strip_id(best_match, return_index=True)
        return self._db[name][i], name

    def normalize(self, imgs: np.ndarray) -> np.ndarray:
        img_norm = self.metadata.get("img_norm", "per_image")
        if img_norm == "per_image":
            # linearly scales x to have mean of 0, variance of 1
            std_adj = np.std(imgs, axis=(1, 2, 3), keepdims=True)
            std_adj = np.maximum(std_adj, 1.0 / np.sqrt(imgs.size / len(imgs)))
            mean = np.mean(imgs, axis=(1, 2, 3), keepdims=True)
            return (imgs - mean) / std_adj
        elif img_norm == "fixed":
            # scales x to [-1, 1]
            return (imgs - 127.5) / 128.0
        else:
            raise ValueError(f"mode '{img_norm}' not recognized")

    def _train_classifier(self):
        try:
            if self.classifier_type == "svm":
                self.classifier = svm.SVC(kernel="linear")
            elif self.classifier_type == "knn":
                self.classifier = neighbors.KNeighborsClassifier()

            all_pairs = self.all_pairs()
            names = list(all_pairs.keys())
            embeds = list(all_pairs.values())
            self.classifier.fit(embeds, names)

        except (AttributeError, ValueError):
            raise ValueError("Current model incompatible with database")
