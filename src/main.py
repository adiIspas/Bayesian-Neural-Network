from src.preparation.clasification_type import Type
from src.run_binary import Binary
from src.run_multi_class import MultiClass

# classification_type = Type.MULTI_CLASS
# classification_type = Type.BINARY
classification_type = Type.ALL

if classification_type == Type.MULTI_CLASS:
    MultiClass.run()
elif classification_type == Type.BINARY:
    Binary.run()
elif classification_type == Type.ALL:
    MultiClass.run()
    Binary.run()
