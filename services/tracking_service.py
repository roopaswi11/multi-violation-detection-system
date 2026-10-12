import numpy as np

class TrackingService:
    """
    Multi-object tracking manager. Associates detections across consecutive video frames
    to maintain persistent track IDs using ByteTrack/IoU tracking logic.
    """
    def __init__(self, max_age=30, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.next_id = 1
        # tracks: {track_id: {'bbox': [x1, y1, x2, y2], 'hits': int, 'age': int, 'class_name': str}}
        self.tracks = {}

    @staticmethod
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0.0, xB - xA)
        interH = max(0.0, yB - yA)
        interArea = interW * interH

        boxAArea = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        denom = boxAArea + boxBArea - interArea
        return interArea / denom if denom > 0 else 0.0

    def update(self, detections):
        """
        Takes current frame detections:
        [{'bbox': [x1, y1, x2, y2], 'class_name': str, 'confidence': float, ...}]
        Returns detections with assigned 'track_id' field.
        """
        # Age existing tracks
        for tid in list(self.tracks.keys()):
            self.tracks[tid]['age'] += 1
            if self.tracks[tid]['age'] > self.max_age:
                del self.tracks[tid]

        if not detections:
            return detections

        assigned_det_indices = set()
        assigned_track_ids = set()

        # Match detections to existing tracks via IoU
        for tid, tinfo in self.tracks.items():
            best_iou = self.iou_threshold
            best_match_idx = -1
            
            for idx, det in enumerate(detections):
                if idx in assigned_det_indices:
                    continue
                score = self.iou(tinfo['bbox'], det['bbox'])
                if score > best_iou:
                    best_iou = score
                    best_match_idx = idx
                    
            if best_match_idx >= 0:
                self.tracks[tid]['bbox'] = detections[best_match_idx]['bbox']
                self.tracks[tid]['hits'] += 1
                self.tracks[tid]['age'] = 0
                detections[best_match_idx]['track_id'] = tid
                assigned_det_indices.add(best_match_idx)
                assigned_track_ids.add(tid)

        # Create new tracks for unassigned detections
        for idx, det in enumerate(detections):
            if idx not in assigned_det_indices:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {
                    'bbox': det['bbox'],
                    'hits': 1,
                    'age': 0,
                    'class_name': det.get('class_name', '')
                }
                det['track_id'] = new_id

        return detections
