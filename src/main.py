from src.preparation.clasification_type import Type
from src.run_binary import Binary
from src.run_multi_class import MultiClass

# classification_type = Type.MULTI_CLASS
classification_type = Type.BINARY

if classification_type == Type.MULTI_CLASS:
    print("\n=== Multi class classification ===\n")
    MultiClass.run()
elif classification_type == Type.BINARY:
    print("\n=== Binary class classification ===\n")
    Binary.run()
