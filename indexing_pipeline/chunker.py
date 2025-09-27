class Chunker:
    """Basic text chunker, by character."""
    
    def chunk_text(self, text: str, chunk_size: int, overlap: int, split_on_whitespace: bool = True) -> list[str]:
        if split_on_whitespace:
            return self._chunk_on_whitespace(text, chunk_size, overlap)
        return self._chunk_naively(text, chunk_size, overlap)
    
    def _chunk_on_whitespace(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        chunks: list[str] = []
        length = len(text)
        index = 0
        
        while index < length:
            prev_whitespace = 0
            left_index = index - overlap
            while left_index >= 0:
                if text[left_index] == " ":
                    prev_whitespace = left_index
                    break
                left_index -= 1
            next_whitespace = text.find(" ", index + chunk_size)
            if next_whitespace == -1:
                next_whitespace = len(text)
            chunk = text[prev_whitespace: next_whitespace].strip()
            chunks.append(chunk)
            index = next_whitespace + 1
        
        return chunks

    def _chunk_naively(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        chunks: list[str] = []
        length = len(text)
        index = 0

        while index < length:        
            start = max(0, index - overlap + 1)
            end = min(index + chunk_size + overlap, len(text))
            chunk = text[start: end].strip() 
            chunks.append(chunk)
            index += chunk_size

        return chunks