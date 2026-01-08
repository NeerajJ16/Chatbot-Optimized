from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    section: str   # e.g. "experience", "skills"
    source: str    # filename
