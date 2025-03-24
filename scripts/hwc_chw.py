#
# Script used to analyze the HWC <-> CHW mapping for designing the image_transit_layout kernel.
# Result: can't do an inplace transition (i.e. image_transit_layout(image) is not possible, copy is needed).
#

W = 16
H = 16
C = 3


def hwc_index(x, y, c):
    return (y * H + x) * C + c


def hwc_index_to_coord(i) -> tuple[int, int, int]:
    y = int(i / (W * C))
    x = int((i / C) % W)
    c = i % C
    return x, y, c


def chw_index(x, y, c):
    return c * H * W + y * H + x


def find_cycle(x, y, c):
    from_ = x, y, c
    cycle = set({})
    while True:
        # Convert HWC coordinates to 1D index
        from_idx = hwc_index(from_[0], from_[1], from_[2])
        if from_idx in cycle:
            break
        cycle.add(from_idx)
        # Convert coordinates to CHW index
        to_ = chw_index(from_[0], from_[1], from_[2])
        from_ = hwc_index_to_coord(to_)
    return frozenset(cycle)


def hwc_to_chw():
    cycles = set()
    for y in range(0, H):
        print(f"Line {y}/{H}...")
        for x in range(0, W):
            for c in range(0, C):
                cycles.add(find_cycle(x, y, c))
    print(f"Found unique {len(cycles)} cycles:")
    for cycle in cycles:
        print(f"  {len(cycle)}: {cycle}")


if __name__ == "__main__":
    hwc_to_chw()
