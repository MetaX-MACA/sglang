
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
System Power Manager for Linux
检查并设置CPU电源模式、PCI ACS功能和GPU电源模式
默认流程：检查所有状态，如不符合预期则自动设置，最后验证设置结果
"""

import os
import sys
import subprocess
import shutil
import json
import argparse
from pathlib import Path

# Debug flag
DEBUG = False

def debug_print(*args, **kwargs):
    """Print debug messages only when DEBUG is True"""
    if DEBUG:
        print("DEBUG:", *args, **kwargs)


def show_table(headers, data, title=None, max_widths=None):
    """Display data in a formatted table with borders and auto-adjusted column widths"""
    if not data:
        print("  No data")
        return

    # Default max widths for each column (None means no limit)
    if max_widths is None:
        max_widths = [None] * len(headers)

    # Calculate column widths
    col_widths = []
    for i, header in enumerate(headers):
        # Start with header width
        max_width = len(str(header))
        # Check all data rows for this column
        for row in data:
            if i < len(row):
                cell_str = str(row[i])
                # Remove newlines and replace with spaces
                cell_str = cell_str.replace('\n', ' ')
                # Apply max width limit if specified
                cell_width = len(cell_str)
                if max_widths[i] is not None and cell_width > max_widths[i]:
                    cell_width = max_widths[i]
                max_width = max(max_width, cell_width)
        # Add some padding
        col_widths.append(max_width + 2)

    # Calculate total table width
    total_width = sum(col_widths) + len(headers) + 1

    # Print title if provided
    if title:
        print(f"  {title}")

    # Print top border
    top_border = "  ┌"
    for width in col_widths:
        top_border += "─" * width + "┬"
    top_border = top_border[:-1] + "┐"
    print(top_border)

    # Print header row
    header_row = "  │"
    for i, header in enumerate(headers):
        header_row += f" {header:<{col_widths[i]-1}}│"
    print(header_row)

    # Print separator line
    separator = "  ├"
    for width in col_widths:
        separator += "─" * width + "┼"
    separator = separator[:-1] + "┤"
    print(separator)

    # Print data rows
    for row in data:
        data_row = "  │"
        for i, cell in enumerate(row):
            if i < len(row):
                cell_str = str(cell)
                # Remove newlines and replace with spaces
                cell_str = cell_str.replace('\n', ' ')
                # Truncate cell content if it exceeds max width
                if max_widths[i] is not None and len(cell_str) > max_widths[i]:
                    cell_str = cell_str[:max_widths[i]-3] + "..."
                data_row += f" {cell_str:<{col_widths[i]-1}}│"
            else:
                data_row += f" {'':<{col_widths[i]-1}}│"
        print(data_row)

    # Print bottom border
    bottom_border = "  └"
    for width in col_widths:
        bottom_border += "─" * width + "┴"
    bottom_border = bottom_border[:-1] + "┘"
    print(bottom_border)


class NetworkInterfaceManager:
    def __init__(self):
        self.network_interfaces_path = "/sys/class/net"

    def get_network_interfaces(self):
        """Get all network interfaces"""
        interfaces = []
        try:
            net_path = Path(self.network_interfaces_path)
            if net_path.exists():
                for interface_dir in net_path.iterdir():
                    if interface_dir.is_dir():
                        interfaces.append(interface_dir.name)
            return sorted(interfaces)
        except Exception as e:
            print(f"获取网络接口列表时出错: {e}")
            return []

    def get_interface_device_id(self, interface):
        """Get PCI device ID for network interface"""
        try:
            # Skip bond and virtual interfaces
            if interface.startswith('bond') or interface == 'lo' or interface == 'docker0':
                return None

            device_link_path = Path(self.network_interfaces_path) / interface / "device"
            if device_link_path.exists():
                # Use readlink to get the actual device path
                result = subprocess.run(
                    ['readlink', '-f', str(device_link_path)],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    return None

                device_path = result.stdout.strip()
                # Extract the actual device ID from the path (the directory name where net/interface exists)
                actual_device_path = Path(device_path)
                if actual_device_path.exists():
                    # Check if this path contains the net directory with our interface
                    net_path = actual_device_path / "net"
                    if net_path.exists():
                        # This is the actual device that owns the network interface
                        device_id = actual_device_path.name
                        if DEBUG:
                            print(f"DEBUG: 接口 {interface} 对应设备ID: {device_id}")
                        return device_id

                # Fallback: try to extract from path using regex
                import re
                match = re.search(r'/([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])/', device_path)
                if match:
                    device_id = match.group(1)
                    if DEBUG:
                        print(f"DEBUG: 接口 {interface} 使用回退设备ID: {device_id}")
                    return device_id
                else:
                    return None
            return None
        except Exception as e:
            if DEBUG:
                print(f"DEBUG: 获取接口 {interface} 设备ID时出错: {e}")
            return None

    def get_interface_model(self, interface):
        """Get network interface model information"""
        try:
            # Check if this is a bond interface
            if interface.startswith('bond'):
                return self._get_bond_interface_model(interface)

            # Try to get device info from various sources
            device_path = Path(self.network_interfaces_path) / interface / "device"
            if device_path.exists():
                # Get PCI device info if available
                vendor_path = device_path / "vendor"
                device_path_file = device_path / "device"

                if vendor_path.exists() and device_path_file.exists():
                    with open(vendor_path, 'r') as f:
                        vendor_id = f.read().strip()
                    with open(device_path_file, 'r') as f:
                        device_id = f.read().strip()

                    # Try to get device name using lspci -mm format
                    try:
                        result = subprocess.run(
                            ["lspci", "-mm", "-d", f"{vendor_id}:{device_id}"],
                            capture_output=True, text=True
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            # Parse lspci -mm output: 6 fields in quotes
                            lines = result.stdout.strip().split('\n')
                            for line in lines:
                                # Split by quotes and filter out empty strings
                                parts = [part.strip('"') for part in line.split('"') if part.strip()]
                                if len(parts) >= 6:
                                    # Use the last non-empty field as model name
                                    # Fields: [pci_address, device_type, vendor, device, subsystem_vendor, subsystem_name]
                                    model_fields = parts[2:6]  # vendor, device, subsystem_vendor, subsystem_name
                                    # Find the last non-empty field
                                    model_name = ""
                                    for field in reversed(model_fields):
                                        if field and field.strip():
                                            model_name = field.strip()
                                            break
                                    if model_name:
                                        return model_name
                    except:
                        pass

                    # If lspci fails, return simplified device info
                    return f"Device {vendor_id}:{device_id}"

            # Try to get driver info
            driver_path = Path(self.network_interfaces_path) / interface / "device" / "driver"
            if driver_path.exists():
                driver_name = driver_path.name
                return f"{driver_name} driver"

            return "Unknown model"

        except Exception as e:
            print(f"获取接口 {interface} 型号时出错: {e}")
            return "Unknown model"

    def _get_bond_interface_model(self, bond_interface):
        """Get bond interface model information including slave interfaces"""
        try:
            # Get bond slave interfaces
            bonding_path = Path(self.network_interfaces_path) / bond_interface / "bonding"
            if not bonding_path.exists():
                return "Bond interface"

            slaves_path = bonding_path / "slaves"
            if slaves_path.exists():
                with open(slaves_path, 'r') as f:
                    slaves = f.read().strip().split()

                if slaves:
                    # Show interface names instead of models
                    if len(slaves) == 1:
                        return f"Bond: {slaves[0]}"
                    else:
                        return f"Bond: {', '.join(slaves[:3])}{'...' if len(slaves) > 3 else ''}"

            return "Bond interface"

        except Exception as e:
            print(f"获取bond接口 {bond_interface} 信息时出错: {e}")
            return "Bond interface"

    def get_interface_link_status(self, interface):
        """Get network interface link status"""
        try:
            # Skip loopback interface and other special interfaces
            if interface == 'lo':
                return {
                    'status': 'loopback',
                    'speed': 'N/A',
                    'duplex': 'N/A'
                }

            operstate_path = Path(self.network_interfaces_path) / interface / "operstate"
            if operstate_path.exists():
                with open(operstate_path, 'r') as f:
                    operstate = f.read().strip()

                # Get speed if available
                speed_path = Path(self.network_interfaces_path) / interface / "speed"
                speed = "Unknown"
                if speed_path.exists():
                    try:
                        with open(speed_path, 'r') as f:
                            speed_value = f.read().strip()
                            if speed_value.isdigit():
                                speed_num = int(speed_value)
                                # Convert to appropriate unit
                                # Note: Linux speed files typically report in Mbps
                                if speed_num >= 1000000:     # 1 Tbps = 1000 Gbps = 1,000,000 Mbps
                                    speed = f"{speed_num / 1000000:.1f} Tbps"
                                elif speed_num >= 1000:      # 1 Gbps = 1000 Mbps
                                    speed = f"{speed_num / 1000:.1f} Gbps"
                                elif speed_num >= 1:         # 1 Mbps = 1 Mbps
                                    speed = f"{speed_num} Mbps"
                                else:
                                    speed = f"{speed_num} Kbps"
                    except (OSError, IOError):
                        speed = "Unknown"

                # Get duplex mode if available
                duplex_path = Path(self.network_interfaces_path) / interface / "duplex"
                duplex = "Unknown"
                if duplex_path.exists():
                    try:
                        with open(duplex_path, 'r') as f:
                            duplex = f.read().strip()
                    except (OSError, IOError):
                        duplex = "Unknown"

                return {
                    'status': operstate,
                    'speed': speed,
                    'duplex': duplex
                }
            else:
                return {
                    'status': 'Unknown',
                    'speed': 'Unknown',
                    'duplex': 'Unknown'
                }

        except Exception as e:
            # Don't print error for every interface, just return unknown status
            return {
                'status': 'Unknown',
                'speed': 'Unknown',
                'duplex': 'Unknown'
            }

    def get_all_interface_status(self):
        """Get status for all network interfaces"""
        interfaces = self.get_network_interfaces()
        interface_status = {}

        for interface in interfaces:
            model = self.get_interface_model(interface)
            link_status = self.get_interface_link_status(interface)

            interface_status[interface] = {
                'model': model,
                'link_status': link_status['status'],
                'speed': link_status['speed'],
                'duplex': link_status['duplex']
            }

        # Sort interfaces by speed (descending order)
        sorted_interfaces = sorted(
            interface_status.items(),
            key=lambda x: self._extract_speed_value(x[1]['speed']),
            reverse=True
        )

        return dict(sorted_interfaces)

    def _extract_speed_value(self, speed_str):
        """Extract numeric speed value from speed string for sorting"""
        try:
            if speed_str == "Unknown" or speed_str == "N/A":
                return 0

            # Handle new format like "200.0 Gbps", "50.0 Gbps", "25.0 Mbps"
            parts = speed_str.split()
            if parts and len(parts) >= 2:
                speed_num = float(parts[0])
                unit = parts[1].lower()

                # Convert to Mbps for consistent sorting
                if unit == "tbps":
                    return int(speed_num * 1000000)  # 1 Tbps = 1,000,000 Mbps
                elif unit == "gbps":
                    return int(speed_num * 1000)     # 1 Gbps = 1,000 Mbps
                elif unit == "mbps":
                    return int(speed_num)            # 1 Mbps = 1 Mbps
                elif unit == "kbps":
                    return int(speed_num / 1000)     # 1 Kbps = 0.001 Mbps
                else:
                    return int(speed_num)

            return 0
        except:
            return 0

    def _format_speed(self, speed_value):
        """Format speed value with appropriate unit (Kbps, Mbps, Gbps, Tbps)"""
        try:
            if speed_value == "Unknown" or speed_value == "N/A":
                return speed_value

            # Extract numeric value if it's in "50000 Mbps" format
            if isinstance(speed_value, str):
                parts = speed_value.split()
                if parts and parts[0].isdigit():
                    speed_num = int(parts[0])
                else:
                    return speed_value
            else:
                speed_num = speed_value

            # Convert to appropriate unit
            if speed_num >= 1000000000:  # 1 Tbps = 1000 Gbps
                return f"{speed_num / 1000000000:.1f} Tbps"
            elif speed_num >= 1000000:   # 1 Gbps = 1000 Mbps
                return f"{speed_num / 1000000:.1f} Gbps"
            elif speed_num >= 1000:      # 1 Mbps = 1000 Kbps
                return f"{speed_num / 1000:.1f} Mbps"
            else:
                return f"{speed_num} Kbps"
        except:
            return speed_value


class NUMAManager:
    def __init__(self):
        self.numa_path = "/sys/devices/system/node"

    def get_numa_nodes(self):
        """Get all available NUMA nodes"""
        try:
            numa_path = Path(self.numa_path)
            if not numa_path.exists():
                return []

            nodes = []
            for node_dir in numa_path.iterdir():
                if node_dir.name.startswith('node') and node_dir.is_dir():
                    node_id = node_dir.name.replace('node', '')
                    if node_id.isdigit():
                        nodes.append(int(node_id))

            return sorted(nodes)
        except Exception as e:
            print(f"获取NUMA节点时出错: {e}")
            return []

    def _format_memory(self, memory_str):
        """Format memory value with appropriate unit (KiB, MiB, GiB, TiB)"""
        try:
            if memory_str == "Unknown":
                return memory_str

            # Extract numeric value and unit from memory string (e.g., "1056497276 kB")
            parts = memory_str.split()
            if len(parts) >= 2:
                memory_num = float(parts[0])
                unit = parts[1].lower()

                # Convert to KiB first for consistent processing
                if unit == "tb" or unit == "tib":
                    memory_kib = memory_num * 1024 * 1024 * 1024
                elif unit == "gb" or unit == "gib":
                    memory_kib = memory_num * 1024 * 1024
                elif unit == "mb" or unit == "mib":
                    memory_kib = memory_num * 1024
                elif unit == "kb" or unit == "kib":
                    memory_kib = memory_num
                else:
                    # Assume it's already in KiB
                    memory_kib = memory_num

                # Convert to appropriate unit
                if memory_kib >= 1024 * 1024 * 1024:  # 1 TiB = 1024 GiB
                    return f"{memory_kib / (1024 * 1024 * 1024):.1f} TiB"
                elif memory_kib >= 1024 * 1024:       # 1 GiB = 1024 MiB
                    return f"{memory_kib / (1024 * 1024):.1f} GiB"
                elif memory_kib >= 1024:              # 1 MiB = 1024 KiB
                    return f"{memory_kib / 1024:.1f} MiB"
                else:
                    return f"{memory_kib:.0f} KiB"

            return memory_str
        except:
            return memory_str

    def get_numa_node_info(self, node_id):
        """Get detailed information for a NUMA node"""
        try:
            node_path = Path(self.numa_path) / f"node{node_id}"
            if not node_path.exists():
                return None

            info = {'node_id': node_id}

            # Get CPU information
            cpu_path = node_path / "cpulist"
            if cpu_path.exists():
                with open(cpu_path, 'r') as f:
                    info['cpulist'] = f.read().strip()

            # Get memory information
            meminfo_path = node_path / "meminfo"
            if meminfo_path.exists():
                with open(meminfo_path, 'r') as f:
                    meminfo = f.read()
                    # Extract total memory
                    for line in meminfo.split('\n'):
                        if 'MemTotal:' in line:
                            parts = line.split()
                            if len(parts) >= 4:
                                raw_memory = f"{parts[3]} {parts[4]}"
                                info['memory_total'] = self._format_memory(raw_memory)
                            break

            # Get distance information
            distance_path = node_path / "distance"
            if distance_path.exists():
                with open(distance_path, 'r') as f:
                    info['distance'] = f.read().strip()

            return info
        except Exception as e:
            print(f"获取NUMA节点 {node_id} 信息时出错: {e}")
            return None

    def check_numa_warning(self, nodes):
        """Check if NUMA nodes count exceeds 4 and issue warning"""
        if len(nodes) > 4:
            print(f"⚠️ 警告: 检测到 {len(nodes)} 个NUMA节点，超过4个节点可能影响性能")
            return True
        return False


class PCIeAPSMManager:
    def __init__(self):
        self.pci_devices_path = "/sys/bus/pci/devices"
        self.target_vendor_id = "9999"

    def get_device_vendor_id(self, device_id):
        """Get device vendor ID"""
        try:
            vendor_path = Path(self.pci_devices_path) / device_id / "vendor"
            if vendor_path.exists():
                with open(vendor_path, 'r') as f:
                    vendor_id = f.read().strip()
                return vendor_id
            return None
        except Exception as e:
            print(f"获取设备 {device_id} vendor ID时出错: {e}")
            return None

    def get_pcie_apsm_status(self):
        """Get PCIe APSM status for vendor id 9999 devices only"""
        try:
            result = subprocess.run(["lspci", "-vv"], capture_output=True, text=True)
            if result.returncode != 0:
                print("获取PCIe设备信息失败")
                return {}

            devices_apsm = {}
            current_device = None
            device_info = {}

            for line in result.stdout.split('\n'):
                line = line.strip()

                # Look for device lines (e.g., "00:00.0 Host bridge: Intel Corporation ...")
                if line and line[0].isdigit() and ':' in line and '.' in line:
                    if current_device and device_info:
                        # Only add device if it has vendor id 9999
                        vendor_id = self.get_device_vendor_id(current_device)
                        if vendor_id and vendor_id.endswith(self.target_vendor_id):
                            devices_apsm[current_device] = device_info

                    current_device = line.split()[0]
                    device_info = {'device_name': ' '.join(line.split()[1:])}

                # Look for LnkCtl line with ASPM information
                elif line.startswith('LnkCtl:'):
                    if 'ASPM' in line:
                        if 'ASPM Disabled' in line:
                            device_info['aspm_status'] = 'Disabled'
                        elif 'ASPM L0s' in line or 'ASPM L1' in line:
                            device_info['aspm_status'] = 'Enabled'
                        else:
                            device_info['aspm_status'] = 'Unknown'

                # Look for LnkCap line with ASPM support information
                elif line.startswith('LnkCap:'):
                    if 'ASPM not supported' in line:
                        device_info['aspm_support'] = 'Not Supported'
                    elif 'ASPM L0s' in line or 'ASPM L1' in line:
                        device_info['aspm_support'] = 'Supported'
                    else:
                        device_info['aspm_support'] = 'Unknown'

            # Add the last device if it has vendor id 9999
            if current_device and device_info:
                vendor_id = self.get_device_vendor_id(current_device)
                if vendor_id and vendor_id.endswith(self.target_vendor_id):
                    devices_apsm[current_device] = device_info

            return devices_apsm

        except Exception as e:
            print(f"获取PCIe APSM状态时出错: {e}")
            return {}


class GPUPowerManager:
    def __init__(self):
        self.mx_smi_cmd = "mx-smi"

    def check_mx_smi_available(self):
        """Check if mx-smi command is available"""
        try:
            result = subprocess.run([self.mx_smi_cmd, "--version"],
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def get_gpu_pcie_status(self):
        """Get GPU PCIe status using mx-smi --show-pcie"""
        try:
            result = subprocess.run([self.mx_smi_cmd, "--show-pcie"],
                                  capture_output=True, text=True)

            # Only show debug info if command failed
            if result.returncode != 0:
                print(f"执行命令: {self.mx_smi_cmd} --show-pcie")
                print(f"命令返回码: {result.returncode}")
                print(f"命令标准输出长度: {len(result.stdout)} 字符")
                print(f"命令错误输出: {result.stderr}")

            gpu_pcie_status = {}

            if result.stdout.strip():
                import re

                # Use regex to parse GPU PCIe information
                # Pattern to match GPU lines: GPU#0  MXC5XX  0000:0e:00.0
                gpu_pattern = r'^GPU#(\d+)\s+\w+\s+([0-9a-fA-F:\.]+)'

                # Pattern to match PCIe info lines with proper indentation
                current_speed_pattern = r'^\s+Current Speed\s+:\s+(.+)$'
                current_width_pattern = r'^\s+Current Width\s+:\s+(.+)$'
                max_speed_pattern = r'^\s+Max Speed\s+:\s+(.+)$'
                max_width_pattern = r'^\s+Max Width\s+:\s+(.+)$'

                lines = result.stdout.split('\n')
                current_gpu = None
                current_pcie_info = {}

                for line in lines:
                    # Check for GPU line
                    gpu_match = re.match(gpu_pattern, line.strip())
                    if gpu_match:
                        # Save previous GPU data if exists
                        if current_gpu is not None and current_pcie_info:
                            gpu_pcie_status[current_gpu] = current_pcie_info

                        current_gpu = gpu_match.group(1)
                        current_pcie_info = {'device_id': gpu_match.group(2)}
                        continue

                    # Check for PCIe info lines
                    if current_gpu is not None:
                        current_speed_match = re.match(current_speed_pattern, line)
                        if current_speed_match:
                            current_pcie_info['current_speed'] = current_speed_match.group(1).strip()
                            continue

                        current_width_match = re.match(current_width_pattern, line)
                        if current_width_match:
                            current_pcie_info['current_width'] = current_width_match.group(1).strip()
                            continue

                        max_speed_match = re.match(max_speed_pattern, line)
                        if max_speed_match:
                            current_pcie_info['max_speed'] = max_speed_match.group(1).strip()
                            continue

                        max_width_match = re.match(max_width_pattern, line)
                        if max_width_match:
                            current_pcie_info['max_width'] = max_width_match.group(1).strip()
                            # Save this GPU data when we get max width (last field)
                            gpu_pcie_status[current_gpu] = current_pcie_info
                            current_pcie_info = {}
                            continue

                # Save the last GPU data if not already saved
                if current_gpu is not None and current_pcie_info and current_gpu not in gpu_pcie_status:
                    gpu_pcie_status[current_gpu] = current_pcie_info

            # Only show parsing result if command failed
            if result.returncode != 0:
                print(f"解析到的GPU PCIe状态: {gpu_pcie_status}")

            return gpu_pcie_status

        except Exception as e:
            print(f"获取GPU PCIe状态时出错: {e}")
            return None

    def get_gpu_info(self):
        """Get GPU information including model and power mode using mx-smi"""
        try:
            result = subprocess.run([self.mx_smi_cmd, "--show-power-mode"],
                                  capture_output=True, text=True)

            # Only show debug info if command failed
            if result.returncode != 0:
                print(f"执行命令: {self.mx_smi_cmd} --show-power-mode")
                print(f"命令返回码: {result.returncode}")
                print(f"命令标准输出长度: {len(result.stdout)} 字符")
                print(f"命令错误输出: {result.stderr}")

            gpu_info = {}

            if result.stdout.strip():
                # Parse the text output to extract GPU information
                lines = result.stdout.split('\n')
                current_gpu = None
                current_model = None

                for line in lines:
                    line = line.strip()

                    # Look for GPU lines (e.g., "GPU#0  MXC5XX  0000:0e:00.0")
                    if line.startswith('GPU#'):
                        parts = line.split()
                        if len(parts) >= 3:
                            current_gpu = parts[0].replace('GPU#', '')  # Extract GPU number
                            current_model = parts[1]  # Extract GPU model

                    # Look for power mode lines (e.g., "mode                                      : Normal")
                    elif line.startswith('mode') and ':' in line and current_gpu is not None:
                        mode_value = line.split(':', 1)[1].strip()
                        gpu_info[current_gpu] = {
                            'model': current_model,
                            'power_mode': mode_value
                        }
                        current_gpu = None  # Reset for next GPU
                        current_model = None

            # Only show parsing result if command failed
            if result.returncode != 0:
                print(f"解析到的GPU信息: {gpu_info}")

            return gpu_info

        except Exception as e:
            print(f"获取GPU信息时出错: {e}")
            return None

    def get_gpu_power_mode(self):
        """Get GPU power mode using mx-smi (backward compatibility)"""
        gpu_info = self.get_gpu_info()
        if gpu_info:
            return {gpu_id: info['power_mode'] for gpu_id, info in gpu_info.items()}
        return None

    def set_gpu_power_mode(self, gpu_id, mode=1):
        """Set GPU power mode using mx-smi"""
        try:
            cmd = [self.mx_smi_cmd, "--set-power-mode", str(mode), "-i", str(gpu_id)]
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            print(f"命令返回码: {result.returncode}")
            print(f"命令标准输出: {result.stdout}")
            print(f"命令错误输出: {result.stderr}")

            if result.returncode == 0:
                print(f"成功: GPU {gpu_id} 电源模式已设置为 {mode}")
                return True
            else:
                print(f"错误: 设置GPU {gpu_id} 电源模式失败")
                print(f"返回码: {result.returncode}")
                print(f"错误输出: {result.stderr}")
                return False
        except Exception as e:
            print(f"设置GPU电源模式时出错: {e}")
            return False


class PCIACSManager:
    def __init__(self):
        self.pci_devices_path = "/sys/bus/pci/devices"
        self.target_vendor_id = "9999"

    def check_p2p_support(self, device_id):
        """Check if PCIe P2P (Peer-to-Peer) is supported and enabled for a device using setpci"""
        try:
            if not device_id or device_id == 'Unknown':
                return "Unknown"

            # Use setpci to read PCI configuration space for P2P related capabilities
            try:
                # Read PCIe Device Capabilities register (offset 0x4)
                result = subprocess.run(
                    ["setpci", "-s", device_id, "CAP_EXP+0x4.l"],
                    capture_output=True, text=True
                )

                if result.returncode == 0:
                    # Parse the capability register value
                    cap_value = result.stdout.strip()
                    if cap_value:
                        # Convert hex string to integer
                        try:
                            cap_int = int(cap_value, 16)
                            # Check for P2P related bits in PCIe Device Capabilities
                            # Bit 6: P2P Enable Supported
                            # Bit 7: P2P Enable
                            p2p_supported = bool(cap_int & (1 << 6))
                            p2p_enabled = bool(cap_int & (1 << 7))

                            if p2p_supported and p2p_enabled:
                                return "P2P Enabled"
                            elif p2p_supported:
                                return "P2P Supported"
                        except ValueError:
                            pass

                # Alternative method: check for SR-IOV capability using setpci
                try:
                    result = subprocess.run(
                        ["setpci", "-s", device_id, "CAP_SRIOV+0x4.l"],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        sriov_value = result.stdout.strip()
                        if sriov_value:
                            try:
                                sriov_int = int(sriov_value, 16)
                                # Check SR-IOV capability
                                if sriov_int & 0x1:  # SR-IOV capability enabled
                                    return "SR-IOV Enabled"
                            except ValueError:
                                pass
                except:
                    pass

                # Fallback to lspci method for SR-IOV detection
                try:
                    result = subprocess.run(
                        ["lspci", "-s", device_id, "-v"],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        output = result.stdout.lower()
                        if "single root i/o virtualization" in output or "sr-iov" in output:
                            return "SR-IOV Supported"
                except:
                    pass

                return "Not supported"

            except Exception as e:
                if DEBUG:
                    print(f"DEBUG: setpci检查设备 {device_id} 失败: {e}")
                return "Unknown"

        except Exception as e:
            if DEBUG:
                print(f"检查设备 {device_id} P2P支持时出错: {e}")
            return "Unknown"

    def check_p2p_connectivity(self, device1_id, device2_id):
        """Check if two devices can communicate via P2P"""
        try:
            if not device1_id or not device2_id or device1_id == 'Unknown' or device2_id == 'Unknown':
                return "Unknown"

            # Check if both devices are in the same PCIe domain and can potentially communicate
            domain1 = device1_id.split(':')[0]
            domain2 = device2_id.split(':')[0]

            if domain1 != domain2:
                return "Different domains"

            # Check if both devices support P2P
            device1_p2p = self.check_p2p_support(device1_id)
            device2_p2p = self.check_p2p_support(device2_id)

            if "P2P Enabled" in device1_p2p and "P2P Enabled" in device2_p2p:
                return "P2P Connected"
            elif "P2P Supported" in device1_p2p and "P2P Supported" in device2_p2p:
                return "P2P Possible"
            else:
                return "No P2P"

        except Exception as e:
            if DEBUG:
                print(f"检查设备 {device1_id} 和 {device2_id} P2P连通性时出错: {e}")
            return "Unknown"

    def check_root_privileges(self):
        """Check if the script is running with root privileges"""
        if os.geteuid() != 0:
            print("错误: 需要root权限来修改PCI ACS设置")
            print("请使用sudo运行此脚本")
            return False
        return True

    def get_pci_devices(self):
        """Get all PCI devices"""
        devices = []
        try:
            pci_path = Path(self.pci_devices_path)
            if pci_path.exists():
                for device_dir in pci_path.iterdir():
                    if device_dir.is_dir():
                        devices.append(device_dir.name)
            return devices
        except Exception as e:
            print(f"获取PCI设备列表时出错: {e}")
            return []

    def get_device_info(self, device_id):
        """Get device vendor and device ID"""
        try:
            vendor_path = Path(self.pci_devices_path) / device_id / "vendor"
            device_path = Path(self.pci_devices_path) / device_id / "device"

            if vendor_path.exists() and device_path.exists():
                with open(vendor_path, 'r') as f:
                    vendor_id = f.read().strip()
                with open(device_path, 'r') as f:
                    device_id_val = f.read().strip()
                return vendor_id, device_id_val
            return None, None
        except Exception as e:
            print(f"获取设备 {device_id} 信息时出错: {e}")
            return None, None

    def check_acs_capability(self, device_id):
        """Check ACS capability using setpci command"""
        try:
            # Use setpci to read ACS register
            cmd = ["setpci", "-s", device_id, "ECAP_ACS+0x6.w"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                acs_value = result.stdout.strip()
                print(f"设备 {device_id} ACS寄存器值: {acs_value}")
                # Check if ACS is enabled (non-zero value means enabled)
                return acs_value != "0000"
            else:
                # Device doesn't have ACS capability, consider it as disabled
                if "not found" in result.stderr:
                    return False
                else:
                    print(f"读取设备 {device_id} ACS寄存器失败: {result.stderr}")
                    return False
        except Exception as e:
            print(f"检查设备 {device_id} ACS功能时出错: {e}")
            return False

    def disable_acs(self, device_id):
        """Disable ACS for a device using setpci command"""
        try:
            # First check if device has ACS capability
            if not self.check_acs_capability(device_id):
                print(f"设备 {device_id} 没有ACS功能，无需禁用")
                return True

            # Use setpci to write 0000 to ACS register
            cmd = ["setpci", "-v", "-s", device_id, "ECAP_ACS+0x6.w=0000"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"已禁用设备 {device_id} 的ACS功能")
                return True
            else:
                print(f"禁用设备 {device_id} ACS功能失败: {result.stderr}")
                return False
        except PermissionError:
            print(f"错误: 没有权限禁用设备 {device_id} 的ACS功能")
            return False
        except Exception as e:
            print(f"禁用设备 {device_id} ACS功能时出错: {e}")
            return False

    def find_target_devices(self):
        """Find devices with vendor id 9999"""
        target_devices = []
        devices = self.get_pci_devices()

        for device_id in devices:
            vendor_id, device_id_val = self.get_device_info(device_id)
            if vendor_id and vendor_id.endswith(self.target_vendor_id):
                acs_enabled = self.check_acs_capability(device_id)
                target_devices.append({
                    'device_id': device_id,
                    'vendor_id': vendor_id,
                    'device_id_val': device_id_val,
                    'acs_enabled': acs_enabled
                })

        return target_devices

    def check_and_disable_acs_for_target(self):
        """Check and disable ACS for vendor id 9999 devices"""
        print("检查PCI ACS状态...")
        target_devices = self.find_target_devices()

        if not target_devices:
            print("未找到vendor id为9999的PCI设备")
            return True

        print(f"找到 {len(target_devices)} 个vendor id为9999的设备:")

        changes_made = False
        for device in target_devices:
            print(f"  设备: {device['device_id']}")
            print(f"    Vendor ID: {device['vendor_id']}")
            print(f"    Device ID: {device['device_id_val']}")
            print(f"    ACS状态: {'启用' if device['acs_enabled'] else '禁用'}")

            if device['acs_enabled']:
                print(f"    正在禁用ACS功能...")
                if self.disable_acs(device['device_id']):
                    changes_made = True
                else:
                    print(f"    禁用ACS失败")
            print()

        return changes_made


class CPUPowerModeManager:
    def __init__(self):
        self.scaling_governor_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        self.available_governors_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors"

    def check_root_privileges(self):
        """Check if the script is running with root privileges"""
        if os.geteuid() != 0:
            print("错误: 需要root权限来修改CPU电源模式")
            print("请使用sudo运行此脚本")
            return False
        return True

    def get_current_power_mode(self):
        """Get the current CPU power mode"""
        try:
            with open(self.scaling_governor_path, 'r') as f:
                current_mode = f.read().strip()
            return current_mode
        except FileNotFoundError:
            print("错误: 无法找到CPU电源管理文件")
            print("可能的原因:")
            print("1. 系统不支持CPU频率调节")
            print("2. cpufreq驱动未加载")
            print("3. 文件路径不存在")
            return None
        except PermissionError:
            print("错误: 没有权限读取CPU电源管理文件")
            return None
        except Exception as e:
            print(f"读取当前电源模式时出错: {e}")
            return None

    def get_available_governors(self):
        """Get available CPU power governors"""
        try:
            with open(self.available_governors_path, 'r') as f:
                governors = f.read().strip().split()
            return governors
        except FileNotFoundError:
            print("错误: 无法找到可用电源模式列表")
            return []
        except Exception as e:
            print(f"获取可用电源模式时出错: {e}")
            return []

    def set_power_mode(self, mode):
        """Set CPU power mode to specified governor"""
        try:
            # Check if the mode is available
            available_modes = self.get_available_governors()
            if mode not in available_modes:
                print(f"错误: 电源模式 '{mode}' 不可用")
                print(f"可用的模式: {', '.join(available_modes)}")
                return False

            # Set the mode for all CPUs
            cpu_dirs = list(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"))
            success_count = 0
            total_cpus = 0

            for cpu_dir in cpu_dirs:
                governor_file = cpu_dir / "cpufreq" / "scaling_governor"
                if governor_file.exists():
                    try:
                        with open(governor_file, 'w') as f:
                            f.write(mode)
                        success_count += 1
                    except Exception as e:
                        print(f"警告: 无法为 {cpu_dir.name} 设置电源模式: {e}")
                    total_cpus += 1

            if success_count == total_cpus:
                print(f"成功: 所有 {total_cpus} 个CPU的电源模式已设置为 '{mode}'")
                return True
            else:
                print(f"部分成功: {success_count}/{total_cpus} 个CPU的电源模式已设置为 '{mode}'")
                return success_count > 0

        except Exception as e:
            print(f"设置电源模式时出错: {e}")
            return False


class SystemPowerManager:
    def __init__(self, show_apsm=False):
        self.cpu_manager = CPUPowerModeManager()
        self.pci_manager = PCIACSManager()
        self.gpu_manager = GPUPowerManager()
        self.pcie_apsm_manager = PCIeAPSMManager()
        self.network_manager = NetworkInterfaceManager()
        self.numa_manager = NUMAManager()
        self.verification_failed = False
        self.show_apsm = show_apsm

    def check_and_install_tools(self):
        """Check and install required tools for CentOS and Ubuntu"""
        print("检查必要工具...")
        required_tools = ['setpci', 'lspci']
        missing_tools = []

        for tool in required_tools:
            try:
                result = subprocess.run(['which', tool], capture_output=True, text=True)
                if result.returncode != 0:
                    missing_tools.append(tool)
            except Exception:
                missing_tools.append(tool)

        if not missing_tools:
            print("✓ 所有必要工具已安装")
            return True

        print(f"缺少必要工具: {', '.join(missing_tools)}")
        print("正在尝试自动安装...")

        # Detect package manager
        package_manager = None
        if subprocess.run(['which', 'apt-get'], capture_output=True).returncode == 0:
            package_manager = 'apt'
        elif subprocess.run(['which', 'yum'], capture_output=True).returncode == 0:
            package_manager = 'yum'
        elif subprocess.run(['which', 'dnf'], capture_output=True).returncode == 0:
            package_manager = 'dnf'

        if not package_manager:
            print("错误: 无法检测到包管理器 (apt-get, yum, dnf)")
            print("请手动安装以下工具:")
            for tool in missing_tools:
                if tool == 'setpci' or tool == 'lspci':
                    print(f"  - {tool} (pciutils 包)")
            return False

        # Map tools to packages
        tool_packages = {
            'setpci': 'pciutils',
            'lspci': 'pciutils'
        }

        packages_to_install = []
        for tool in missing_tools:
            if tool in tool_packages:
                packages_to_install.append(tool_packages[tool])

        # Remove duplicates
        packages_to_install = list(set(packages_to_install))

        if not packages_to_install:
            print("错误: 无法确定需要安装的包")
            return False

        print(f"使用 {package_manager} 安装包: {', '.join(packages_to_install)}")

        try:
            if package_manager == 'apt':
                subprocess.run(['apt-get', 'update', '-y'], capture_output=True, text=True)
                result = subprocess.run(
                    ['apt-get', 'install', '-y'] + packages_to_install,
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    print("apt 安装失败，尝试切换为 MetaX 内网镜像后重试...")
                    if self._switch_to_metax_apt_mirror():
                        subprocess.run(['apt-get', 'update', '-y'], capture_output=True, text=True)
                        result = subprocess.run(
                            ['apt-get', 'install', '-y'] + packages_to_install,
                            capture_output=True, text=True,
                        )
            else:
                result = subprocess.run(
                    [package_manager, 'install', '-y'] + packages_to_install,
                    capture_output=True, text=True,
                )

            if result.returncode == 0:
                print("✓ 工具安装成功")
                return True
            else:
                print(f"工具安装失败: {result.stderr}")
                return False

        except subprocess.CalledProcessError as e:
            print(f"工具安装失败: {e}")
            return False
        except Exception as e:
            print(f"工具安装时出错: {e}")
            return False

    def _switch_to_metax_apt_mirror(self):
        """将 apt 源切换为 MetaX 内网镜像 repo.metax-tech.com，返回是否成功。

        仅在默认 apt 源不可达时调用；会备份原配置。
        """
        mirror = "http://repo.metax-tech.com/r/ubuntu"
        codename = None
        try:
            with open("/etc/os-release", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VERSION_CODENAME="):
                        codename = line.split("=", 1)[1].strip().strip('"')
                        break
        except Exception:
            pass
        if not codename:
            try:
                codename = subprocess.run(
                    ["lsb_release", "-cs"], capture_output=True, text=True
                ).stdout.strip() or None
            except Exception:
                codename = None
        if not codename:
            print("警告: 无法识别发行版代号，跳过 apt 源切换")
            return False

        lines = [
            f"deb {mirror}/ {codename} main restricted universe multiverse",
            f"deb {mirror}/ {codename}-updates main restricted universe multiverse",
            f"deb {mirror}/ {codename}-backports main restricted universe multiverse",
            f"deb {mirror}/ {codename}-security main restricted universe multiverse",
        ]
        try:
            src = "/etc/apt/sources.list"
            if os.path.exists(src):
                shutil.copy2(src, src + ".bak.envchk")
            with open(src, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            src_d = "/etc/apt/sources.list.d"
            if os.path.isdir(src_d):
                for name in os.listdir(src_d):
                    if name.endswith((".list", ".sources")) and not name.endswith(".bak.envchk"):
                        try:
                            os.rename(os.path.join(src_d, name), os.path.join(src_d, name + ".bak.envchk"))
                        except Exception:
                            pass
            print(f"已将 apt 源切换为内网镜像: {mirror} (codename={codename})")
            return True
        except Exception as e:
            print(f"切换 apt 源失败: {e}")
            return False

    def check_all(self):
        """Check all system power settings"""
        print("=== 系统电源状态检查 ===")

        # Check CPU power mode for all CPUs
        print("\nCPU电源状态:")
        cpu_dirs = list(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"))
        cpu_ok = True
        cpu_count = 0
        performance_count = 0
        non_performance_count = 0
        non_performance_cpus = []

        for cpu_dir in cpu_dirs:
            governor_file = cpu_dir / "cpufreq" / "scaling_governor"
            if governor_file.exists():
                try:
                    with open(governor_file, 'r') as f:
                        cpu_mode = f.read().strip()
                    cpu_count += 1

                    if cpu_mode == "performance":
                        performance_count += 1
                    else:
                        non_performance_count += 1
                        non_performance_cpus.append(cpu_dir.name)
                        cpu_ok = False
                except Exception as e:
                    print(f"  {cpu_dir.name}: 读取失败 - {e}")
                    cpu_ok = False

        cpu_checked = cpu_count > 0
        if not cpu_checked:
            print("  ⚠️ 未检测到 CPU 调频信息 (容器内 /sys/.../cpufreq 可能不可见)，CPU 电源模式无法验证")

        # Compressed display for large number of CPUs
        if cpu_count <= 20:
            # Show all CPUs if count is small
            for cpu_dir in cpu_dirs:
                governor_file = cpu_dir / "cpufreq" / "scaling_governor"
                if governor_file.exists():
                    try:
                        with open(governor_file, 'r') as f:
                            cpu_mode = f.read().strip()
                        status = "✓" if cpu_mode == "performance" else "✗"
                        print(f"  {cpu_dir.name}: {cpu_mode} {status}")
                    except Exception:
                        pass
        else:
            # Show summary for large number of CPUs
            print(f"  ✓ performance模式: {performance_count} 个CPU")
            if non_performance_count > 0:
                print(f"  ✗ 非performance模式: {non_performance_count} 个CPU")
                # Show first 5 non-performance CPUs as examples
                if len(non_performance_cpus) > 5:
                    print(f"    示例: {', '.join(non_performance_cpus[:5])} 等")
                else:
                    print(f"    具体: {', '.join(non_performance_cpus)}")
            else:
                print(f"  ✓ 所有CPU均为performance模式")

        print(f"总计: {cpu_count} 个CPU核心")

        # Check GPU power mode, PCIe status and PCI ACS status in one table
        print("\nGPU Device Status:")
        gpu_info_data = self.gpu_manager.get_gpu_info()
        gpu_pcie_data = self.gpu_manager.get_gpu_pcie_status()
        target_devices = self.pci_manager.find_target_devices()

        # Create device mapping for quick lookup
        device_acs_map = {}
        for device in target_devices:
            device_acs_map[device['device_id']] = device['acs_enabled']

        gpu_ok = True
        pci_ok = True

        if gpu_info_data and isinstance(gpu_info_data, dict) and gpu_info_data:
            # Prepare data for table
            headers = ["GPU", "Model", "Power Mode", "PCIe Link", "ACS Status", "P2P Status"]
            table_data = []

            for gpu_id, gpu_info in gpu_info_data.items():
                power_mode = gpu_info['power_mode']
                model = gpu_info['model']

                # Get PCIe info
                pcie_info = gpu_pcie_data.get(gpu_id, {}) if gpu_pcie_data else {}
                device_id = pcie_info.get('device_id', 'Unknown')
                current_speed = pcie_info.get('current_speed', '?')
                current_width = pcie_info.get('current_width', '?')
                pcie_link = f"{current_speed} {current_width}"

                # Get ACS status
                acs_enabled = device_acs_map.get(device_id, False)
                acs_status = "✓ Disabled" if not acs_enabled else "✗ Enabled"

                # Get P2P status
                p2p_status = self.pci_manager.check_p2p_support(device_id) if device_id != 'Unknown' else "Unknown"

                # Convert power mode text to numeric value and status
                if power_mode.lower() == "high":
                    numeric_mode = "1"
                    power_display = f"High({numeric_mode})✓"
                elif power_mode.lower() == "normal":
                    numeric_mode = "0"
                    power_display = f"Normal({numeric_mode})✗"
                else:
                    numeric_mode = "Unknown"
                    power_display = f"{power_mode}({numeric_mode})?"

                table_data.append([
                    gpu_id,
                    model,
                    power_display,
                    pcie_link,
                    acs_status,
                    p2p_status
                ])

                # Check if power mode is not 1 (we want mode 1)
                if numeric_mode != "1":
                    gpu_ok = False

                # Check if ACS is enabled (we want it disabled)
                if acs_enabled:
                    pci_ok = False

            # Set max widths for columns: [GPU, Model, Power Mode, PCIe Link, ACS Status, P2P Status]
            max_widths = [4, 10, 15, 15, 12, 12]
            show_table(headers, table_data, f"Total: {len(gpu_info_data)} GPUs", max_widths)
        else:
            print("  Unable to get valid GPU information")
            gpu_ok = True  # Consider GPU check passed if mx-smi not available

        # Check PCIe APSM status (only show if requested)
        if self.show_apsm:
            print("\nPCIe APSM状态:")
            apsm_data = self.pcie_apsm_manager.get_pcie_apsm_status()
            apsm_summary = self._summarize_apsm_status(apsm_data)

            print(f"  APSM支持设备: {apsm_summary['supported']}")
            print(f"  APSM启用设备: {apsm_summary['enabled']}")
            print(f"  APSM禁用设备: {apsm_summary['disabled']}")
            print(f"  APSM未知状态设备: {apsm_summary['unknown']}")
            print(f"  总计PCIe设备: {apsm_summary['total']}")

        # Check P2P connectivity between GPUs and network interfaces
        print("\nP2P连通性检查:")
        gpu_pcie_data = self.gpu_manager.get_gpu_pcie_status()
        network_status = self.network_manager.get_all_interface_status()
        if gpu_pcie_data and network_status:
            # Get GPU device IDs
            gpu_devices = []
            for gpu_id, pcie_info in gpu_pcie_data.items():
                device_id = pcie_info.get('device_id', 'Unknown')
                if device_id != 'Unknown':
                    gpu_devices.append((gpu_id, device_id))

            # Get network interface device IDs (skip bond and virtual interfaces)
            net_devices = []
            for interface in network_status:
                if interface.startswith('bond') or interface == 'docker0' or interface == 'lo':
                    continue
                device_id = self.network_manager.get_interface_device_id(interface)
                if device_id:
                    net_devices.append((interface, device_id))

            # Create connectivity matrix
            if gpu_devices and net_devices:
                # Prepare matrix data
                headers = ["GPU/Network"] + [f"GPU{gpu_id}" for gpu_id, _ in gpu_devices]
                matrix_data = []

                # For each network interface, show connectivity to all GPUs
                for net_interface, net_device in net_devices:
                    row = [net_interface]
                    for gpu_id, gpu_device in gpu_devices:
                        connectivity = self.pci_manager.check_p2p_connectivity(gpu_device, net_device)
                        # Convert connectivity status to ✓/✗ symbols
                        if connectivity == "P2P Connected":
                            row.append("✓")
                        elif connectivity == "P2P Possible":
                            row.append("?")
                        else:
                            row.append("✗")
                    matrix_data.append(row)

                # Show matrix table
                show_table(headers, matrix_data, "GPU与网络接口P2P连通性矩阵", max_widths=[15] + [15] * len(gpu_devices))
            else:
                print("  无法获取GPU或网络接口设备信息")
        else:
            print("  无法获取GPU或网络接口信息进行P2P连通性检查")

        # Check network interface status
        print("\n网络接口状态:")

        if network_status:
            # Prepare data for table
            headers = ["Interface", "Model", "Link Status", "Speed", "Duplex", "P2P Status"]
            table_data = []

            for interface, info in network_status.items():
                # Skip loopback interface
                if interface == 'lo':
                    continue

                # Simplify link status display
                if info['link_status'] == 'up':
                    link_status = "✓"
                elif info['link_status'] == 'down':
                    link_status = "✗"
                else:
                    link_status = f"? {info['link_status']}"

                # Get P2P status for physical interfaces
                device_id = self.network_manager.get_interface_device_id(interface)
                p2p_status = self.pci_manager.check_p2p_support(device_id) if device_id else "N/A"

                table_data.append([
                    interface,
                    info['model'],
                    link_status,
                    info['speed'],
                    info['duplex'],
                    p2p_status
                ])

            # Set max widths for columns: [Interface, Model, Link Status, Speed, Duplex, P2P Status]
            max_widths = [12, 40, 10, 12, 10, 12]
            show_table(headers, table_data, f"Total: {len(table_data)} network interfaces", max_widths)
        else:
            print("  无法获取网络接口信息")

        # Check NUMA nodes
        print("\nNUMA节点状态:")
        numa_nodes = self.numa_manager.get_numa_nodes()
        if numa_nodes:
            print(f"  检测到 {len(numa_nodes)} 个NUMA节点: {numa_nodes}")

            # Show detailed NUMA information
            numa_table_data = []
            for node_id in numa_nodes:
                node_info = self.numa_manager.get_numa_node_info(node_id)
                if node_info:
                    cpu_info = node_info.get('cpulist', 'Unknown')
                    memory_info = node_info.get('memory_total', 'Unknown')
                    numa_table_data.append([
                        f"Node {node_id}",
                        cpu_info,
                        memory_info
                    ])

            if numa_table_data:
                headers = ["NUMA Node", "CPU List", "Memory"]
                show_table(headers, numa_table_data, max_widths=[10, 30, 15])

            # Check for NUMA warning
            numa_warning = self.numa_manager.check_numa_warning(numa_nodes)
        else:
            print("  未检测到NUMA节点")
            numa_warning = False

        print(f"\n检查结果:")
        print(f"  CPU: {'✓ 符合要求' if (cpu_ok and cpu_checked) else ('✗ 需要设置' if cpu_checked else '⚠️ 无法验证')}")
        print(f"  PCI ACS: {'✓ 符合要求' if pci_ok else '✗ 需要设置'}")
        print(f"  GPU: {'✓ 符合要求' if gpu_ok else '✗ 需要设置'}")
        if self.show_apsm:
            print(f"  PCIe APSM: 信息汇总 (见上方)")
        print(f"  网络接口: 信息汇总 (见上方)")
        print(f"  NUMA: {'⚠️ 警告: 节点数过多' if numa_warning else '✓ 正常'}")

        return cpu_ok, pci_ok, gpu_ok

    def _summarize_apsm_status(self, apsm_data):
        """Summarize PCIe APSM status"""
        summary = {
            'total': len(apsm_data),
            'supported': 0,
            'enabled': 0,
            'disabled': 0,
            'unknown': 0
        }

        for device_id, device_info in apsm_data.items():
            # Count ASPM support
            if device_info.get('aspm_support') == 'Supported':
                summary['supported'] += 1

            # Count ASPM status
            status = device_info.get('aspm_status', 'Unknown')
            if status == 'Enabled':
                summary['enabled'] += 1
            elif status == 'Disabled':
                summary['disabled'] += 1
            else:
                summary['unknown'] += 1

        return summary

    def set_all(self):
        """Set all system power settings to optimal values"""
        print("\n=== 系统电源设置 ===")

        changes_made = False

        # Set CPU power mode
        current_cpu_mode = self.cpu_manager.get_current_power_mode()
        if current_cpu_mode and current_cpu_mode != "performance":
            if self.cpu_manager.check_root_privileges():
                print("设置CPU电源模式为performance...")
                if self.cpu_manager.set_power_mode("performance"):
                    changes_made = True
            else:
                print("跳过CPU设置: 需要root权限")

        # Set PCI ACS
        target_devices = self.pci_manager.find_target_devices()
        if target_devices:
            for device in target_devices:
                if device['acs_enabled']:
                    if self.pci_manager.check_root_privileges():
                        print(f"禁用设备 {device['device_id']} 的ACS功能...")
                        if self.pci_manager.disable_acs(device['device_id']):
                            changes_made = True
                    else:
                        print("跳过PCI ACS设置: 需要root权限")
                        break

        # Set GPU power mode
        gpu_data = self.gpu_manager.get_gpu_power_mode()
        if gpu_data and isinstance(gpu_data, dict) and gpu_data:
            for gpu_id, power_mode in gpu_data.items():
                # Convert power mode text to numeric value
                # "High" represents 1, "Normal" represents 0
                # We need to set to 1 if current mode is not 1
                if power_mode.lower() != "high":  # Current mode is not 1, need to set to 1
                    print(f"设置GPU {gpu_id} 电源模式为1...")
                    if self.gpu_manager.set_gpu_power_mode(gpu_id, 1):
                        changes_made = True

        return changes_made

    def verify_all(self):
        """Verify all settings are correct"""
        print("\n=== 设置验证 ===")

        cpu_ok, pci_ok, gpu_ok = self.check_all()

        if not cpu_ok:
            print("✗ CPU电源模式验证失败")
            self.verification_failed = True

        if not pci_ok:
            print("✗ PCI ACS验证失败")
            self.verification_failed = True

        if not gpu_ok:
            print("✗ GPU电源模式验证失败")
            self.verification_failed = True

        # PCIe APSM and network interfaces are informational only, don't affect verification
        if self.show_apsm:
            print("✓ PCIe APSM状态已汇总 (仅信息性)")
        print("✓ 网络接口状态已汇总 (仅信息性)")

        if not self.verification_failed:
            print("✓ 所有设置验证通过")

        return not self.verification_failed

    def run(self):
        """Main workflow: check tools -> check -> set -> verify"""
        print("系统电源管理器启动...")
        debug_print("开始执行run方法")

        # Step 0: Check and install required tools
        debug_print("开始工具检查")
        if not self.check_and_install_tools():
            print("错误: 必要工具检查失败，无法继续")
            return False
        debug_print("工具检查完成")

        # Step 1: Check current status
        debug_print("开始系统状态检查")
        cpu_ok, pci_ok, gpu_ok = self.check_all()
        debug_print(f"系统状态检查完成 - CPU={cpu_ok}, PCI={pci_ok}, GPU={gpu_ok}")

        # If all are already optimal, show status and exit successfully
        if cpu_ok and pci_ok and gpu_ok:
            print("\n✓ 所有系统电源设置已符合要求")
            print("系统状态:")
            print("  - CPU: 所有核心均为performance模式")
            print("  - PCI ACS: vendor id 9999设备ACS已禁用")
            print("  - GPU: 所有GPU均为High模式(1)")
            print("  - 网络接口: 状态已汇总")
            debug_print("所有设置符合要求，程序结束")
            return True

        debug_print("需要设置，继续执行设置流程")
        # Step 2: Apply settings
        changes_made = self.set_all()

        if changes_made:
            # Step 3: Verify settings
            verification_ok = self.verify_all()

            if verification_ok:
                print("\n✓ 系统电源设置完成")
                return True
            else:
                print("\n✗ 设置验证失败，部分设置可能未生效")
                return False
        else:
            print("\n! 未进行任何设置更改")
            return True


def main():
    global DEBUG
    import sys

    parser = argparse.ArgumentParser(
        description="系统电源管理器 - 检查并设置CPU、PCI ACS、GPU电源模式、PCIe APSM状态和网络接口状态",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
系统电源管理器完整说明
========================

功能概述
--------
本程序是一个全面的Linux系统电源管理工具，能够自动检查和优化以下系统设置：
- CPU电源模式 (performance/powersave)
- PCI ACS功能 (vendor id 9999设备)
- GPU电源模式 (High/Normal)
- PCIe APSM状态 (仅vendor id 9999设备)
- 网络接口状态 (型号和链路状态)

工作流程
--------
1. 工具检查阶段
   - 自动检测系统包管理器 (apt-get/yum/dnf)
   - 检查必要工具 (setpci, lspci) 是否已安装
   - 自动安装缺失的pciutils包

2. 状态检查阶段
   - 检查所有CPU核心的电源模式
   - 检查vendor id 9999设备的PCI ACS状态
   - 检查所有GPU的电源模式
   - 检查vendor id 9999设备的PCIe APSM状态并生成汇总
   - 检查所有网络接口的型号和链路状态

3. 自动设置阶段 (需要root权限)
   - 将CPU电源模式设置为performance
   - 禁用vendor id 9999设备的ACS功能
   - 将GPU电源模式设置为High (模式1)

4. 验证阶段
   - 验证所有设置是否生效
   - 提供详细的验证结果

系统要求
--------
- 操作系统: CentOS 7+, Ubuntu 16.04+, 或其他兼容Linux发行版
- 权限要求: 设置功能需要root权限
- 必要工具: setpci, lspci (自动安装)
- GPU支持: 需要mx-smi工具 (沐曦GPU)

使用示例
--------
# 仅检查系统状态，不进行任何设置
%(prog)s --check-only

# 执行完整工作流程 (需要root权限)
sudo %(prog)s

# 查看详细帮助信息
%(prog)s --help

退出代码说明
------------
0: 所有设置符合要求或设置成功
1: 检查发现需要设置的项目 (仅检查模式)
2: 设置验证失败

注意事项
--------
- 程序会自动处理工具依赖，无需手动安装
- 兼容CentOS和Ubuntu系统
- 设置PCI ACS和CPU电源模式需要root权限
- GPU设置需要mx-smi工具支持
- PCIe APSM状态仅检查vendor id 9999设备，为信息性汇总，不影响设置结果
- 网络接口状态为信息性汇总，显示所有接口的型号、链路状态、速度和双工模式
- 默认不显示PCIe APSM信息，使用--apsm参数启用显示
        """
    )

    parser.add_argument(
        '--check-only',
        action='store_true',
        help='仅检查系统状态，不进行任何设置'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式，显示详细执行信息'
    )

    parser.add_argument(
        '--apsm',
        action='store_true',
        help='显示PCIe APSM状态信息'
    )

    args = parser.parse_args()

    # Set debug flag
    DEBUG = args.debug
    debug_print("进入main函数")

    manager = SystemPowerManager(show_apsm=args.apsm)

    if args.check_only:
        # Only check status
        cpu_ok, pci_ok, gpu_ok = manager.check_all()
        if cpu_ok and pci_ok and gpu_ok:
            print("\n✓ 所有系统电源设置符合要求")
            sys.exit(0)
        else:
            print("\n! 部分设置需要调整")
            sys.exit(1)
    else:
        # Full workflow: check -> set -> verify
        success = manager.run()
        if success:
            sys.exit(0)
        else:
            sys.exit(2)  # Exit code 2 for verification failure


if __name__ == "__main__":
    main()


