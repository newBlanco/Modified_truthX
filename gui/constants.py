from enum import IntEnum
import os

# For the gradio web server
SERVER_ERROR_MSG = (
    "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
)
MODERATION_MSG = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE FIX YOUR INPUT AND TRY AGAIN."
CONVERSATION_LIMIT_MSG = "YOU HAVE REACHED THE CONVERSATION LENGTH LIMIT. PLEASE CLEAR HISTORY AND START A NEW CONVERSATION."
INPUT_CHAR_LEN_LIMIT = 2560
CONVERSATION_LEN_LIMIT = 50
LOGDIR = "."

# For the controller and workers(could be overwritten through ENV variables.)
CONTROLLER_HEART_BEAT_EXPIRATION = int(
    os.getenv("CONTROLLER_HEART_BEAT_EXPIRATION", 90)
)
WORKER_HEART_BEAT_INTERVAL = int(os.getenv("WORKER_HEART_BEAT_INTERVAL", 30))
WORKER_API_TIMEOUT = int(os.getenv("WORKER_API_TIMEOUT", 20))


class ErrorCode(IntEnum):
    """
    https://platform.openai.com/docs/guides/error-codes/api-errors
    """

    VALIDATION_TYPE_ERROR = 40001

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103

    INVALID_MODEL = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
    GRADIO_REQUEST_ERROR = 50003
    GRADIO_STREAM_UNKNOWN_ERROR = 50004
    CONTROLLER_NO_WORKER = 50005
    CONTROLLER_WORKER_TIMEOUT = 50006
