import unittest
import torch
from pytorch_lit import PartialLoader
import os


class PartialLoadingTest(unittest.TestCase):
    def test_read(self):
        t = torch.rand((200, 1, 10), dtype=torch.float16)
        torch.save({
            "name": t
        }, ".tmp_file")
        loader = PartialLoader(".tmp_file")
        t2 = loader.read_key("name")
        self.assertTrue(torch.equal(t, t2))
        loader.close()
        os.remove(".tmp_file")


if __name__ == '__main__':
    unittest.main()
