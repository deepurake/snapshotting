#!/usr/bin/env python3
"""Emit criu --external mnt[...] flags for every bind mount in a process's
mount namespace whose source lives outside the container's own overlay
rootfs (NVIDIA driver files injected by nvidia-container-toolkit, plus
Docker's per-container /etc/hosts, /etc/hostname, /etc/resolv.conf, plus
any other host-file bind mounts like -v volumes).

Usage:
    gen_external_mounts.py <pid> dump    > dump_args.txt
    gen_external_mounts.py <pid> restore > restore_args.txt

One argv token per line; read into a bash array with `mapfile -t`.
"""
import sys

VIRTUAL_FSTYPES = {"proc", "sysfs", "cgroup", "cgroup2", "devpts", "mqueue", "tmpfs", "shm"}


def main():
    pid, mode = sys.argv[1], sys.argv[2]
    with open(f"/proc/{pid}/mountinfo") as f:
        lines = f.readlines()

    root_dev = next(
        fields[2] for fields in (line.split() for line in lines) if fields[4] == "/"
    )

    tag_n = 0
    for line in lines:
        fields = line.split()
        mount_source_root, mountpoint, dev = fields[3], fields[4], fields[2]
        fstype = fields[fields.index("-") + 1]

        if mountpoint == "/" or dev == root_dev or fstype in VIRTUAL_FSTYPES:
            continue

        tag = f"ext{tag_n}"
        tag_n += 1
        rel_mountpoint = "." + mountpoint

        if mode == "dump":
            print(f"mnt[{rel_mountpoint}]:{tag}")
        else:
            print(f"mnt[{tag}]:{mount_source_root}")


if __name__ == "__main__":
    main()
