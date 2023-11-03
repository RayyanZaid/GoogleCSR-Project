WINDOW_SIZE : int = 100
MOMENT_SIZE : int = 24


import traceback 

# ensures that all functions print errors and continue execution
def print_error_and_continue(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()  # print error
            print(f"An error occurred in {func.__name__}: {str(e)}")
    return wrapper