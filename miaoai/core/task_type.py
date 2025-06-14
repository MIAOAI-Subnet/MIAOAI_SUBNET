from enum import Enum

class TaskType(Enum):

    TEXT_CLASSIFICATION = "text_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    SCENE_UNDERSTANDING = "scene_understanding"
    OBJECT_DETECTION = "object_detection"
    EMOTION_ANALYSIS = "emotion_analysis"
    
    @classmethod
    def from_str(cls, category: str) -> "TaskType":

        try:
            return cls(category)
        except ValueError:
            return cls.TEXT_CLASSIFICATION
            
    def get_response_format(self) -> dict:
        formats = {
            self.TEXT_CLASSIFICATION: {
                "required": ["class_id", "confidence"],
                "optional": ["description"],
                "example": {
                    "class_id": 1,
                    "confidence": 0.95,
                    "description": "Class 1"
                }
            },
            self.SENTIMENT_ANALYSIS: {
                "required": ["sentiment_score", "confidence"],
                "optional": ["description"],
                "example": {
                    "sentiment_score": 0.8,
                    "description": "Positive",
                    "confidence": 0.9
                }
            },
            self.SCENE_UNDERSTANDING: {
                "required": ["description", "confidence"],
                "optional": ["tags", "entities"],
                "example": {
                    "description": "A sunny beach with people",
                    "confidence": 0.85,
                    "tags": ["beach", "sunny", "people"]
                }
            },
            self.OBJECT_DETECTION: {
                "required": ["objects"],
                "optional": ["confidence", "bounding_boxes"],
                "example": {
                    "objects": ["car", "person"],
                    "confidence": 0.9,
                    "bounding_boxes": [[100, 100, 200, 200]]
                }
            },
            self.EMOTION_ANALYSIS: {
                "required": ["description", "confidence"],
                "optional": ["emotion_intensity"],
                "example": {
                    "description": "Happy",
                    "confidence": 0.85,
                    "emotion_intensity": 0.7
                }
            }
        }
        return formats[self] 