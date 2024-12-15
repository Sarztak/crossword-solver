import easyocr
import cv2


def preprocess_image(image_path):
    """Preprocess the image to improve OCR results."""
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binarization
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Optional: Resize for better OCR
    resized = cv2.resize(binary, (binary.shape[1] * 2, binary.shape[0] * 2), interpolation=cv2.INTER_LINEAR)

    return resized


def extract_text_with_easyocr(image_path):
    """Extract text from the image using EasyOCR."""
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # Perform OCR
    results = reader.readtext(preprocessed_image, detail=0)  # Extract text only
    return "\n".join(results)


if __name__ == "__main__":
    image_path = "images/crossword_sample.jpg"  # Replace with your image path

    # Extract text
    extracted_text = extract_text_with_easyocr(image_path)

    print("Extracted Text:")
    print(extracted_text)
