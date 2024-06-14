import os
from typing import List, Dict
from unstructured.partition.pdf import partition_pdf


class PDFLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def _elements_to_text(self, elements: List) -> str:
        return "\n\n".join(str(el) for el in elements)

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        elements = partition_pdf(self.path)

        self.documents.append(
            {
                "text": self._elements_to_text(elements),
                "metadata": {
                    "filename": os.path.basename(self.path),
                },
            }
        )

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".pdf"):
                    elements = partition_pdf(os.path.join(root, file))

                    self.documents.append(
                        {
                            "text": self._elements_to_text(elements),
                            "metadata": {
                                "filename": file,
                            },
                        }
                    )

    def load_documents(self):
        self.load()
        return self.documents


class CharacterPDFSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[Dict[str, any]]) -> List[Dict[str, any]]:
        chunks = []
        for text in texts:
            split_chunks = self.split(text["text"])

            for i, chunk in enumerate(split_chunks):
                chunks.append(
                    {
                        "text": chunk,
                        "metadata": {
                            "filename": text["metadata"]["filename"],
                            "chunk_id": i,
                        },
                    }
                )

        return chunks


if __name__ == "__main__":
    loader = PDFLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterPDFSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
