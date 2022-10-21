import numpy as np


class MdAnalysis:

    def __init__(self, file, format):
        assert format in ('dftbplus', 'tbmalt', 'aims'), ''

        self.trj_data, self.element = getattr(MdAnalysis, format)(self, file)

    def dftbplus(self, file):
        data_dict = {}
        with open(file, 'r') as f:
            lines = f.readlines()

        n_atoms = int(lines[0])
        assert len(lines) % (n_atoms + 2) == 0
        n_trj = int(len(lines) // (n_atoms + 2))
        trj = []

        for i_trj in range(n_trj):
            start = int(i_trj * (n_atoms + 2))
            end = start + n_atoms
            rows = lines[start: end]

            idata = [list(map(float, row.split()[1: 4])) for row in rows[2:]]

            if i_trj == 0:
                element =[row.split()[0] for row in rows[2:]]

            # data_dict.update({i_trj: np.asarray(idata)})
            trj.append(np.asarray(idata))

        return np.stack(trj), element

    def aims(self, file):
        data_dict = {}
        with open(file, 'r') as f:
            lines = f.readlines()

        n_atoms = int(lines[0])
        assert len(lines) % (n_atoms + 2) == 0
        n_trj = int(len(lines) // (n_atoms + 2))
        trj = []

        for i_trj in range(n_trj):
            start = int(i_trj * (n_atoms + 2))
            end = start + n_atoms
            rows = lines[start: end]

            idata = [list(map(float, row.split()[1: 4])) for row in rows[2:]]

            if i_trj == 0:
                element =[row.split()[0] for row in rows[2:]]

            # data_dict.update({i_trj: np.asarray(idata)})
            trj.append(np.asarray(idata))

        return np.stack(trj), element

    def tbmalt(self):
        pass

    def transfer_geometry(self):
        pass

    def _get_conductivity(self):
        pass

    def _get_slope(self):
        pass

    def RDF(self, cutoff=10.0):
        rdf = hist / (4 * np.pi * edges[1:]**2 * bin_width * len(positions_1))

    def plot_RDF(self):
        pass


if __name__ == '__main__':
    # trj = Trajectory('/Users/gz_fan/Downloads/software/dftbplus/dftbplus/work/battery/lipscl/Li5PS4Cl2/md/geo_end.xyz', 'dftbplus')
    trj = MdAnalysis('/Users/gz_fan/Documents/ML/li5_1000.pos.xyz', 'dftbplus')
    print('trj_data', trj.trj_data.shape)
    msd = ((trj.trj_data[1:] - trj.trj_data[0]) ** 2).sum(-1).sum(-1)
    print(msd.shape)

    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(msd)), msd)
    plt.show()

    plt.plot(np.arange(1000), msd[:1000])
    plt.show()
