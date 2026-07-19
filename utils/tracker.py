"""
utils/tracker.py

Lightweight, dependency-free tracker based on Intersection over Union (IoU)
for tracking multiple currency notes across frames.
"""

class TrackedNote:
    def __init__(self, note_id, denom, conf, bbox, start_time):
        """
        Initialize a tracked note.
        
        Args:
            note_id (int): Unique identifier for this note.
            denom (str): Denomination string (e.g. '500').
            conf (float): Detection confidence score.
            bbox (list): Bounding box [x1, y1, x2, y2].
            start_time (float): Timestamp when the note was first seen.
        """
        self.id = note_id
        self.denom = denom
        self.conf = conf
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.centroid = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        self.start_time = start_time
        self.last_seen = start_time
        self.frames_active = 1
        self.frames_lost = 0
        self.counted = False

    def update(self, denom, conf, bbox, current_time):
        """Update note state with new detection data."""
        self.bbox = bbox
        self.centroid = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        self.last_seen = current_time
        self.frames_lost = 0
        self.frames_active += 1
        
        # If the denomination changes, update it if the new detection has higher confidence
        if denom != self.denom and conf > self.conf:
            self.denom = denom
            self.conf = conf


class NoteTracker:
    def __init__(self, iou_threshold=0.25, max_lost_frames=8, min_active_frames=3,
                 min_active_seconds=0.0):
        """
        Initialize the note tracker.
        
        Args:
            iou_threshold (float): Minimum IoU to match a detection to a tracked note.
            max_lost_frames (int): Number of consecutive frames a note can go missing before deletion.
            min_active_frames (int): Number of consecutive active frames before the note is officially counted.
            min_active_seconds (float): Minimum time a note must remain visible
                before it is considered stable. Set to 0 to disable this check.
        """
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.min_active_frames = min_active_frames
        self.min_active_seconds = min_active_seconds
        self.next_id = 1
        self.tracked_notes = {}  # dict of id -> TrackedNote

    def _calculate_iou(self, boxA, boxB):
        """Calculate the Intersection over Union (IoU) of two bounding boxes."""
        # Determine coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        # Area of intersection
        interArea = max(0, xB - xA) * max(0, yB - yA)
        
        # Area of both bounding boxes
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        # Area of union
        unionArea = boxAArea + boxBArea - interArea
        
        if unionArea == 0:
            return 0.0
            
        return interArea / unionArea

    def update(self, detections, current_time):
        """
        Update the tracker with the latest frame's detections.
        
        Args:
            detections (list): List of detections in format (denom, conf, bbox=[x1, y1, x2, y2]).
            current_time (float): Current system timestamp.
            
        Returns:
            list: TrackedNote instances that have just become stable and should be counted.
        """
        matched_detections = set()
        matched_tracked = set()
        
        # Calculate IoU matches
        matches = []
        for tid, tracked in self.tracked_notes.items():
            for idx, (denom, conf, bbox) in enumerate(detections):
                iou = self._calculate_iou(tracked.bbox, bbox)
                if iou >= self.iou_threshold:
                    matches.append((iou, tid, idx))
                    
        # Sort matches by highest IoU first
        matches.sort(key=lambda x: x[0], reverse=True)
        
        # Match detections to existing tracked notes
        for iou, tid, idx in matches:
            if tid in matched_tracked or idx in matched_detections:
                continue
                
            denom, conf, bbox = detections[idx]
            self.tracked_notes[tid].update(denom, conf, bbox, current_time)
            
            matched_tracked.add(tid)
            matched_detections.add(idx)
            
        # Create new tracked notes for unmatched detections
        for idx, (denom, conf, bbox) in enumerate(detections):
            if idx not in matched_detections:
                new_note = TrackedNote(self.next_id, denom, conf, bbox, current_time)
                self.tracked_notes[self.next_id] = new_note
                self.next_id += 1
                
        # Handle lost notes
        lost_ids = []
        for tid, tracked in list(self.tracked_notes.items()):
            if tid not in matched_tracked:
                tracked.frames_lost += 1
                if tracked.frames_lost > self.max_lost_frames:
                    lost_ids.append(tid)
                    
        for tid in lost_ids:
            del self.tracked_notes[tid]
            
        # Identify notes that are now stable and ready to be counted
        newly_counted = []
        for tid, tracked in self.tracked_notes.items():
            stable_for_long_enough = (
                current_time - tracked.start_time >= self.min_active_seconds
            )
            if (not tracked.counted
                    and tracked.frames_active >= self.min_active_frames
                    and stable_for_long_enough):
                newly_counted.append(tracked)
                
        return newly_counted

    def clear(self):
        """Clear all active tracked notes."""
        self.tracked_notes.clear()
        self.next_id = 1
