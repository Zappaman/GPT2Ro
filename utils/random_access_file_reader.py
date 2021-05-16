import mmap


class RandomAccessFileReader():
    """
    Class that allows random access of a txt file's lines
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.d = []
        self.filelen = 0

        with open(filename, 'r') as f:
            byte_count = 0
            for i, l in enumerate(f):
                if i % 100000 == 0:
                    print(f"Processed {i} lines")
                self.d += [byte_count]
                # assuming utf-8 encoding here
                byte_count += len(l.encode('utf-8'))
                self.filelen += 1
            self.d += [byte_count]

        fd = open(filename, 'r+b')
        self.mmaped_file = mmap.mmap(fd.fileno(), 0)

    def getLine(self, ix: int):
        assert ix >= 0 and ix < self.filelen, "Invalid index!"
        return self.mmaped_file[self.d[ix]:self.d[ix+1]].decode('utf-8').strip()
