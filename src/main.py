from src.preparation.clasification_type import Type
from src import run_multi_class
from src import run_binary

classification_type = Type.MULTI_CLASS
# classification_type = Type.BINARY

if classification_type == Type.MULTI_CLASS:
    print("Multi class classification\n")
    run_multi_class
elif classification_type == Type.BINARY:
    print("Binary class classification\n")
    run_binary
