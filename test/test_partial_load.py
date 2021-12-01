import unittest
import torch
from pytorch_lit import PartialLoader
import os


class PartialLoadingTest(unittest.TestCase):
    def test_read(self):
        f16_tensor = torch.rand((200, 1, 10), dtype=torch.float16)
        torch.save({
            "f16_tensor": f16_tensor,
        }, ".tmp_file")
        loader = PartialLoader(".tmp_file")
        f16_tensor_rcv = loader.read_key("f16_tensor")
        self.assertTrue(torch.equal(f16_tensor, f16_tensor_rcv))
        loader.close()
        os.remove(".tmp_file")

    def test_mixed_types(self):
        f16_tensor = torch.rand((200, 1, 10), dtype=torch.float16)
        int_tensor = torch.randint(0, 1000, (200, 1, 10), dtype=torch.int)
        torch.save({
            "f16_tensor": f16_tensor,
            "int_tensor": int_tensor,
        }, ".tmp_file")
        loader = PartialLoader(".tmp_file")
        f16_tensor_rcv = loader.read_key("f16_tensor")
        self.assertTrue(torch.equal(f16_tensor, f16_tensor_rcv))
        self.assertTrue(f16_tensor.dtype == f16_tensor_rcv.dtype)
        int_tensor_rcv = loader.read_key("int_tensor")
        self.assertTrue(torch.equal(int_tensor, int_tensor_rcv))
        self.assertTrue(int_tensor.dtype == int_tensor_rcv.dtype)
        loader.close()
        os.remove(".tmp_file")

    def test_keys(self):
        f16_tensor = torch.rand((200, 1, 10), dtype=torch.float16)
        int_tensor = torch.randint(0, 1000, (200, 1, 10), dtype=torch.int)
        state_dict = {
            "f16_tensor": f16_tensor,
            "int_tensor": int_tensor,
        }
        torch.save(state_dict, ".tmp_file")
        loader = PartialLoader(".tmp_file")
        self.assertEqual(set(loader.keys()), set(state_dict.keys()))
        loader.close()
        os.remove(".tmp_file")


if __name__ == '__main__':
    unittest.main()
