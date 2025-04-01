# Fingerprint Recognition Server

A Python-based HTTP server that provides fingerprint recognition and management capabilities. This server allows you to register, compare, and manage fingerprint data for different houses using deep learning models.

## Features

- Fingerprint registration and storage
- Fingerprint comparison and matching
- House management with multiple fingerprints
- Support for multiple deep learning models
- Fingerprint enhancement using Gabor filters
- RESTful API endpoints

## Prerequisites

- Python 3.x
- TensorFlow
- OpenCV (cv2)
- NumPy
- Matplotlib
- Fingerprint Enhancer

## Project Structure

```
.
├── src/
│   ├── server.py         # Main server implementation
│   └── util.py           # Utility functions
├── database/
│   ├── database.json     # JSON database for storing house and fingerprint information
│   ├── fingerprints/     # Directory storing fingerprint images
│   └── models/          # Directory containing trained deep learning models
└── README.md
```

## API Endpoints

### POST Endpoints

- `/register_house`: Register a new house with fingerprint data
- `/find_fingerprint`: Search for a matching fingerprint in the database
- `/delete_house`: Remove a house and its associated fingerprints
- `/update_house`: Update house information or fingerprint data

### GET Endpoints

- `/get_fingerprint`: Retrieve fingerprint data for a specific ID

## Usage

1. Start the server:
```bash
python src/server.py
```

2. The server will run on port 8000 by default.

3. Send HTTP requests to the appropriate endpoints with the required data in JSON format.

## Database Structure

The system uses a JSON database to store:
- House information (ID, name)
- Associated fingerprint IDs
- Fingerprint images (stored as .bmp files)

## Models

The server supports multiple deep learning models for fingerprint recognition. Models are loaded from the `database/models` directory and can be selected during fingerprint comparison.

## Security

- Minimum score threshold for fingerprint matching: 0.94
- Fingerprint images are stored in a secure format
- Input validation and error handling implemented

## Error Handling

The server includes comprehensive error handling for:
- Invalid input data
- File operations
- Model loading and inference
- Database operations

## Response Format

All API responses follow a consistent format:
```json
{
    "code": <status_code>,
    "data": <response_data>,
    "message": <optional_message>
}
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
