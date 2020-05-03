import ctypes
import os


class FeedbackSetLib:
    def __init__(self, args):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            self.lib = ctypes.CDLL(
                '{}/build/lib_feedback_set.so'.format(dir_path))
        except OSError:
            self.lib = ctypes.CDLL(
                '{}/build/lib_feedback_set.dylib'.format(dir_path))

        self.lib.Test.restype = ctypes.c_int
        self.lib.TestByMCTS.restype = ctypes.c_int
        self.lib.Train.restype = ctypes.c_float

        arr = (ctypes.c_char_p * len(args))()
        for i in range(len(args)):
            arr[i] = args[i].encode('utf-8')
        self.lib.Init(len(args), arr)

    def __CtypeNetworkX(self, g):
        edges = g.edges()
        e_list_from = (ctypes.c_int * len(edges))()
        e_list_to = (ctypes.c_int * len(edges))()

        if len(edges):
            a, b = zip(*edges)
            e_list_from[:] = a
            e_list_to[:] = b

        return (len(g.nodes()),
                len(edges),
                ctypes.cast(e_list_from, ctypes. c_void_p),
                ctypes.cast(e_list_to, ctypes.c_void_p))

    def SetCurrentGraph(self, g):
        num_nodes, num_edegs, edges_from, edges_to = self.__CtypeNetworkX(g)
        self.lib.SetCurrentGraph(num_nodes, num_edegs, edges_from, edges_to)

    def SetCurrentTestGraph(self, g):
        num_nodes, num_edegs, edges_from, edges_to = self.__CtypeNetworkX(g)
        self.lib.SetCurrentTestGraph(
            num_nodes, num_edegs, edges_from, edges_to)

    def SaveModel(self, filename):
        self.lib.SaveModel(ctypes.c_char_p(filename.encode('utf-8')))

    def LoadModel(self, filename, from_save_dir=True):
        self.lib.LoadModel(ctypes.c_char_p(
            filename.encode('utf-8')), from_save_dir)

    # return solution size
    def Test(self):
        return self.lib.Test()

    def TestByMCTS(self):
        return self.lib.TestByMCTS()

    # return cummulative loss
    def Train(self):
        return self.lib.Train()

    def GenerateTrainData(self, filename):
        self.lib.GenerateTrainData(ctypes.c_char_p(filename.encode('utf-8')))

    def ClearTrainData(self):
        self.lib.ClearTrainData()

    def AddTrainData(self, filename):
        self.lib.AddTrainData(ctypes.c_char_p(filename.encode('utf-8')))
