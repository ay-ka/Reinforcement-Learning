class SampleError(BaseException):
    
    def __init__(self, discription):
        
        super(SampleError, self).__init__(discription)