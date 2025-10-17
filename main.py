import argparse
import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader("file.pdf")
    return " ".join(page.extract_text() or "" for page in reader.pages).strip()

def vectorize_text(text: str, method: str):
    if method.lower() == "tfidf":
        vec = TfidfVectorizer(stop_words="english")
        X = vec.fit_transform([text])
        return X.toarray(), list(vec.get_feature_names_out()) 
    elif method.lower() == "sbert":
        model = SentenceTransformer("all-MiniLM-L6-v2")
        X = model.encode([text], normalize_embeddings=True)
        return np.array(X), []
    else:
        raise ValueError("Method must be 'tfidf' or 'sbert'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to PDF file")
    parser.add_argument("--method", choices=["tfidf", "sbert"], default="tfidf", help="Vectorization method")
    args = parser.parse_args()

    text = extract_text_from_pdf(args.file)
    vectors, vocab = vectorize_text(text, args.method)

    print(f"\nFile: {args.file}")
    print(f"Method: {args.method.upper()}")
    print(f"Vector shape: {vectors.shape}")

    if args.method == "tfidf":
        non_zero = [(w, v) for w, v in zip(vocab, vectors[0]) if v > 0]
        print("Top words:", [w for w, _ in non_zero[:20]])
    else:
        print("First 10 vector dims:", vectors[0][:20])