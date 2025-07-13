# RAG API Logging System

## Overview

The RAG API includes a comprehensive logging system that provides real-time debugging and monitoring capabilities. All logs are automatically generated and stored in the `logs/` directory with different levels of detail for different purposes.

## Log Files

The logging system creates the following log files in the `logs/` directory:

### 1. `debug.log`
- **Purpose**: Contains all debug messages and detailed information
- **Level**: DEBUG and above
- **Content**: Function names, line numbers, detailed execution flow
- **Rotation**: 10MB max size, keeps 5 backup files
- **Use Case**: Detailed debugging and development

### 2. `errors.log`
- **Purpose**: Contains only error messages
- **Level**: ERROR and above
- **Content**: Error details with full stack traces
- **Rotation**: 5MB max size, keeps 3 backup files
- **Use Case**: Error monitoring and troubleshooting

### 3. `api_requests.log`
- **Purpose**: API request and response logging
- **Level**: INFO and above
- **Content**: Request details, response times, status codes
- **Rotation**: 10MB max size, keeps 5 backup files
- **Use Case**: API performance monitoring and usage analytics

## Logging Levels

### DEBUG
- Detailed information for debugging
- Function entry/exit points
- Variable values and data previews
- Example: `"Generated session ID: abc123"`

### INFO
- General information about program execution
- Step-by-step process tracking
- Performance metrics and timing
- Example: `"Training completed successfully in 0.600s"`

### WARNING
- Unexpected situations that don't stop execution
- Deprecated features or configurations
- Example: `"Invalid client secret key provided"`

### ERROR
- Errors that prevent normal operation
- Full stack traces included
- Example: `"Failed to save document to MongoDB"`

## Log Format

### Standard Format
```
2024-01-15 10:30:45,123 - rag_api - INFO - API: Starting training for user: user123
```

### Detailed Format (debug.log)
```
2024-01-15 10:30:45,123 - rag_api - INFO - train:45 - API: Starting training for user: user123
```

## Component-Specific Logging

### API Layer (`main.py`)
- Request/response timing
- Authentication verification
- Error handling and status codes
- Prefix: `API:`

### Router Layer (`train_router.py`, `predict_router.py`)
- Process step tracking
- Component interaction timing
- Data flow monitoring
- Prefix: `ROUTER:`

### Database Layer (`mongo_utils.py`)
- Database operation timing
- Query execution details
- Connection status
- Prefix: `DB:`

### Service Layer (various services)
- External API calls
- Processing steps
- Performance metrics
- Prefix: `SERVICE:`

### Embedding Layer (`embedding_client.py`)
- Embedding API calls and responses
- Text-to-vector conversion timing
- Batch vs single text processing
- Performance metrics and dimensions
- Prefix: `EMBEDDING:`

## Usage Examples

### Basic Logging
```python
from utils.logger import api_logger

# Different log levels
api_logger.debug("Detailed debug information")
api_logger.info("General information")
api_logger.warning("Warning message")
api_logger.error("Error message", exc_info=True)
```

### API Request Logging
```python
api_logger.log_api_request(
    method="POST",
    endpoint="/train",
    user_id="user123",
    session_id="session456",
    file_size=1024
)
```

### Performance Logging
```python
start_time = time.time()
# ... perform operation ...
response_time = time.time() - start_time
api_logger.info(f"Operation completed in {response_time:.3f}s")
```

### Embedding Logging
```python
# Single text embedding
api_logger.info(f"EMBEDDING: Starting single text embedding | Text length: {len(text)} | Model: {self.model}")

# Batch text embedding
api_logger.info(f"EMBEDDING: Starting batch text embedding | Texts count: {len(texts)} | Model: {self.model}")
api_logger.debug(f"EMBEDDING: Total text length: {total_text_length} characters")

# Vector storage
api_logger.info(f"SERVICE: Starting embedding and storage process | Chunks: {len(texts)} | Collection: {self.collection_name}")
```

## Monitoring and Debugging

### Real-time Monitoring
- Logs are written in real-time
- Console output shows INFO level and above
- File logs include all levels

### Performance Analysis
- Timing breakdowns for each operation
- Component-specific performance metrics
- Bottleneck identification

### Error Tracking
- Full stack traces for errors
- Error context and user information
- Error frequency and patterns

### API Analytics
- Request/response patterns
- User activity tracking
- Performance trends

## Configuration

### Log Levels
- Console: INFO (configurable)
- Debug file: DEBUG
- Error file: ERROR
- API file: INFO

### File Rotation
- Automatic rotation based on file size
- Configurable backup count
- Prevents disk space issues

### Customization
Edit `utils/logger.py` to modify:
- Log formats
- File sizes and rotation
- Log levels
- Output destinations

## Testing the Logging System

Run the test script to see the logging system in action:

```bash
python test_logging_demo.py
```

This will generate sample logs demonstrating all logging features.

### Testing Embedding Logging

To specifically test the enhanced embedding logging functionality:

```bash
python test_embedding_logging.py
```

This will demonstrate:
- Document embedding during training
- Query embedding during prediction
- Detailed timing breakdowns
- Performance metrics for embedding operations

## Best Practices

1. **Use appropriate log levels**
   - DEBUG for detailed debugging
   - INFO for general flow
   - WARNING for potential issues
   - ERROR for actual errors

2. **Include context**
   - User IDs, session IDs
   - File names, operation types
   - Timing information

3. **Performance logging**
   - Time critical operations
   - Include timing breakdowns
   - Monitor for bottlenecks

4. **Error handling**
   - Always include `exc_info=True` for errors
   - Provide meaningful error messages
   - Include relevant context

5. **Security**
   - Don't log sensitive data
   - Be careful with user input in logs
   - Consider log file permissions

## Troubleshooting

### Common Issues

1. **Logs not appearing**
   - Check if `logs/` directory exists
   - Verify file permissions
   - Check log level configuration

2. **Large log files**
   - Check rotation settings
   - Monitor disk space
   - Adjust file size limits

3. **Performance impact**
   - Use appropriate log levels
   - Avoid logging in tight loops
   - Consider async logging for high-volume operations

### Log Analysis

Use standard Unix tools to analyze logs:

```bash
# View recent errors
tail -f logs/errors.log

# Search for specific user
grep "user123" logs/debug.log

# Count API requests
grep "API Request" logs/api_requests.log | wc -l

# Find slow operations
grep "completed in" logs/debug.log | grep -E "[0-9]+\.[0-9]{3}s"
``` 