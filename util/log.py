import datetime


class Logger:

    def __init__(self, frame_limit: int = 10, frame_threshold: int = 5):
        # NOTE: Array for matches, sorted chronologically.
        # if len > frame_limit, remove index 0
        self.matches = [""] * frame_limit
        self.frame_limit = frame_limit
        self.frame_threshold = frame_threshold
        self._log = {}

    def log(self, best_match: str):
        self.matches.append(best_match)

        # NOTE: Remove index 0 which is chronologically last frame
        if len(self.matches) > self.frame_limit:
            self.matches.pop(0)

        # dict{match : frequency}
        match_frequencies = {}

        # NOTE: Iteration from most recent frame -> last
        for i in range(self.frame_limit - 1, -1, -1):
            current_match = self.matches[i]
            # NOTE: If match id doesn't exist, set frequency to 1
            if current_match not in match_frequencies.keys():
                match_frequencies.update({current_match: 1})

            # NOTE: Increment frequency
            else:
                match_frequencies.update(
                    {current_match: match_frequencies.get(current_match) + 1}
                )
                # NOTE: if frequency == threshold, detect
                if match_frequencies.get(current_match) >= self.frame_threshold:
                    self._log[datetime.datetime.now()] = current_match
