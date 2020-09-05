class Xyz:
    def __init__(self, x,y):     # Constructor (dunderscore)
        self.x = x
        self.y = y

    def instance_methods(self):     # Methods to do something
        pass

    def __str__(self):
        return 'Type what you want to see when you ask for what the object is'

    def __add__(self, other):       # Add points or objects 
        pass

    def __sub__(self, other):       # Subtract points or objects
        pass


'''
    if we assign one instance method to some variable, we can later
    call other instance methods on that particular variable.
'''


class Abc(Xyz):
    def __init__(self, x, y, z):
        Xyz.__init__(self, x, y)
        self.z = z
        self.p = 0

    def other_instance(self):
        pass



pqr = Abc(x,y,z)
pqr.instance_methods()      # from class Abc
pqr.other_instance()        # from inherited class Xyz
