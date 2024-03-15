from xai_components.base import InArg, OutArg, InCompArg, Component, xai_component

@xai_component
class HelloComponentLibrary(Component):
    input_str: InArg[str]

    def execute(self, ctx) -> None:
        input_str = self.input_str.value
        print("Hello, " + str(input_str))


@xai_component
class InitComponentExample(Component):

    input_str: InArg[str]
    input_int: InArg[int]

    # method 1 
    def __init__(self):
        super().__init__()
        self.input_str.value = "Default Value!"

    def execute(self, ctx) -> None:

        input_str = self.input_str.value

        # method 2
        input_int = self.input_int.value if self.input_int.value else 554
        
        
        print("Hello " + input_str)
        print("Hello " + str(input_int))