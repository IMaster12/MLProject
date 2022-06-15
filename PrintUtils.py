from colorama import init as __colorama_init, Fore as Color, Style
import time
__colorama_init()

__TIME_FORMAT = '{time} | {prefix}{message}' + Color.RESET
__FORMAT = '{prefix}{message}' + Color.RESET
__TIMEFORMAT = '%d-%m-%Y %H:%M:%S'

def printmsg(message, prefix='', show_time=True):
    """
    Print a message with the given prefix

    :param message: The message to print
    :param prefix: The prefix to print
    :param show_time: Show the current time in the message
    :return: None
    """
    if show_time:
        print(__TIME_FORMAT.format(time=time.strftime(__TIMEFORMAT), prefix=prefix, message=message))
    else:
        print(__FORMAT.format(prefix=prefix, message=message))

def getinput(message):
    """
    Ask the user for an input

    :param message: The message to print
    :return: The chosen input
    """
    inputmsg(message)
    return input('Enter: ')

def chooseinputs(message, options):
    """
    Ask the user to choose one of the given options

    :param message: The message to print
    :param options: A list containing the answer options
    :return: The chosen input
    """
    inputmsg(message)
    inp = getinput('[' + ', '.join(options) + ']')
    while inp.lower() not in map(str.lower, options):
        error('Invalid input', show_time=False)
        inputmsg(message)
        inp = getinput('[' + ', '.join(options) + ']')
    return inp

def inputmsg(message):
    """
    Print a message in the input format

    :param message: The message to print
    :return: None
    """
    printmsg(message, show_time=False, prefix=str(Color.BLUE + '' + Style.BRIGHT))

def debug(message, show_time=True):
    """
    Print a debug message

    :param message: The message to print
    :param show_time: Show the current time in the message
    :return: None
    """
    printmsg(message, str(Color.CYAN + '[DEBUG] '), show_time=show_time)

def info(message, show_time=True):
    """
    Print an info message

    :param message: The message to print
    :param show_time: Show the current time in the message
    :return: None
    """
    printmsg(message, str(Color.LIGHTYELLOW_EX + '[INFO] '), show_time=show_time)

def warning(message, show_time=True):
    """
    Print a warning message

    :param message: The message to print
    :param show_time: Show the current time in the message
    :return: None
    """
    printmsg(message, str(Color.YELLOW + '[WARNING] '), show_time=show_time)

def error(message, show_time=True):
    """
    Print a non critical error message

    :param message: The message to print
    :param show_time: Show the current time in the message
    :return: None
    """
    printmsg(message, str(Color.LIGHTRED_EX + '[ERROR] '), show_time=show_time)

def critical(message, show_time=True):
    """
    Print a critical error message

    :param message: The message to print
    :param show_time: Show the current time in the message
    :return: None
    """
    printmsg(message, str(Color.RED + '[CRITICAL] '), show_time=show_time)


# debug("this will get printed")
# info("this will get printed")
# warning("this will get printed")
# error("this will get printed")
# critical("this will get printed")
# getinput("this will get printed")
# chooseinputs('Where do u want it to download from?', ['Github', 'Local'])

# def colored(r, g, b, text):
#     return f"\033[38;2;{r};{g};{b}m{text}\033[38;2;255;255;255m"

