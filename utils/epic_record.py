from .video_record import VideoRecord


class EpicVideoRecord(VideoRecord):
    def __init__(self, tup, rgb4e):
        self._index = str(tup[0])
        self._series = tup[1]
        self.rgb4e = rgb4e

    @property
    def uid(self):
        return self._series['uid']

    @property
    def untrimmed_video_name(self):
        return self._series['video_id']

    @property
    def kitchen(self):
        return int(self._series['video_id'].split('_')[0][1:])

    @property
    def kitchen_p(self):
        return self._series['video_id'].split('_')[0]

    @property
    def recording(self):
        return int(self._series['video_id'].split('_')[1])

    @property
    def start_frame(self):
        return self._series['start_frame'] - 1

    @property
    def end_frame(self):
        return self._series['stop_frame'] - 2

    @property
    def num_frames(self):
        #return self.end_frame - self.start_frame
        return {'RGB': self.end_frame - self.start_frame,
                'Flow': int((self.end_frame - self.start_frame) / 2),
                'Event': int((self.end_frame - self.start_frame) / self.rgb4e),
                'Spec': self.end_frame - self.start_frame}
    @property
    def label(self):
        if 'verb_class' in self._series.keys().tolist():
            label = {'verb': self._series['verb_class'],
                     'noun': self._series['noun_class'] if 'noun_class' in self._series.keys().tolist() else -10000}
        else:  # Fake label to deal with the test sets (S1/S2) that dont have any labels
            label = -10000
        return label
