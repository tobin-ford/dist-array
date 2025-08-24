| Week | Goal                                                                 |
| ---- | -------------------------------------------------------------------- |
| 1    | Serialize/deserialize NumPy arrays with metadata; local test harness |
| 2    | Implement TCP socket server + client that can exchange messages      |
| 3    | Add basic RPC protocol (task name + args)                            |
| 4    | Create a local worker pool that handles array math ops               |
| 5    | Plug in CuPy-based matrix ops                                        |
| 6    | Benchmark latency/performance vs baseline (NumPy, Dask)              |
| 7    | Add batching, basic profiling, or trace logging                      |
| 8+   | Optional: add load balancing, concurrency tuning, real-time metrics  |


We will try to de-couple the application level RPC from the transport layer. 

Eventually, we want to be able to choose whether to send work to CPU, GPU or FPGA accelerator.

The protocol will define how we
1. serialize requests (op codes)
2. serialize data
3. route tasts
4. handle responses, errors and retries

Framing (Length Prefixed)
| Field       | Size    | Type   | Description                                       |
| ----------- | ------- | ------ | ------------------------------------------------- |
| `total_len` | 4 bytes | uint32 | Total message size (header + payload)             |
| `msg_type`  | 1 byte  | uint8  | Request = 0x01, Response = 0x02, Error = 0x03     |
| `flags`     | 1 byte  | uint8  | Target hints: 0x01 = CPU, 0x02 = GPU, 0x04 = FPGA |
| `opcode`    | 2 bytes | uint16 | Operation code (e.g., `MATMUL=0x0001`)            |
| `seq_id`    | 4 bytes | uint32 | Sequence number for request/response pairing      |

Payload Format (Self Describing)
| Field   | Size     | Type      | Description                               |
| ------- | -------- | --------- | ----------------------------------------- |
| `ndim`  | 1 byte   | uint8     | Number of dimensions                      |
| `shape` | 4 × ndim | uint32\[] | Shape of the array                        |
| `dtype` | 1 byte   | uint8     | Enum (e.g., `0x01=float32`, `0x02=int64`) |
| `data`  | N bytes  | raw       | Flat binary array data                    |

Transport Layer Notes
```
class Transport:
    def send(self, data: bytes) -> None: ...
    def recv(self) -> bytes: ...
    def close(self): ...
```

The class will implement TCPTransport, UDPTransport and PCIeTransport under the hood for CPU, FPGA and GPU resepectively.


