import zipfile
import pickle
import torch
import torch._utils
import io


class PartialLoader:
    tensor_func = None

    def __init__(self, path):
        self.path = path
        self._exits = []
        self._read_meta()
        self._prepare_file()

    # noinspection PyProtectedMember
    def _read_meta(self):
        archive = zipfile.ZipFile(self.path, 'r')
        meta_bytes = archive.read('archive/data.pkl')
        meta_buf = io.BytesIO(meta_bytes)
        unpickler = pickle.Unpickler(meta_buf)
        unpickler.persistent_load = PartialLoader._persistent_loader
        PartialLoader.tensor_func = torch._utils._rebuild_tensor
        torch._utils._rebuild_tensor = PartialLoader._rebuild_tensor
        self.meta = unpickler.load()
        torch._utils._rebuild_tensor = PartialLoader.tensor_func
        PartialLoader.tensor_func = None
        archive.close()

    # noinspection PyProtectedMember
    def _prepare_file(self):
        f_open = torch.serialization._open_file_like
        z_open = torch.serialization._open_zipfile_reader
        f_ctx = f_open(self.path, 'rb')
        self._exits.append(f_ctx)
        file = f_ctx.__enter__()
        z_ctx = z_open(file)
        self._exits.append(z_ctx)
        self.zf = z_ctx.__enter__()

    def close(self):
        for e in self._exits:
            e.__exit__()

    def read_key(self, key, device="cpu"):
        info = self.meta[key].tolist()
        print(info)
        _id, num_el, storage_offset, size_l = info[:4]
        size = info[4:4+size_l]
        stride_l = info[4+size_l]
        stride = info[5+size_l:5+size_l+stride_l]
        size = tuple(size)
        stride = tuple(stride)
        storage = self.zf.get_storage_from_record(f'data/{_id}', num_el, torch.float16).storage()
        t = torch.tensor([], dtype=torch.float16, device=device)
        t.set_(storage, storage_offset, size, stride)
        return t

    @staticmethod
    def _persistent_loader(saved_id):
        assert isinstance(saved_id, tuple)
        st = torch.LongStorage()
        setattr(st, "__t", saved_id)
        return st

    @staticmethod
    def _rebuild_tensor(st, storage_offset, size, stride):
        _id = getattr(st, "__t")
        data = [int(_id[2]), int(_id[4]), int(storage_offset), len(size)]
        for x in size:
            data.append(int(x))
        data.append(len(stride))
        for x in stride:
            data.append(int(x))
        t = torch.tensor(data, dtype=st.dtype, device=st.device)
        return t

