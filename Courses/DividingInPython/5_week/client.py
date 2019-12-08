import socket
import time


class Client:
    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = socket.create_connection((self.host, self.port))

    def put(self, metric, value, timestamp=str(int(time.time()))):
        request = bytes(f'put {metric} {value} {timestamp}\n', encoding='utf-8')
        try:
            self.sock.sendall(request)
            while True:
                try:
                    data = self.sock.recv(1024)
                except ClientError:
                    continue

                if not data:
                    continue

                if data.decode("utf-8") == "ok\n\n":
                    break
                elif data.decode("utf-8").startswith("error"):
                    raise ClientError

        except ClientError:
            print("Метрика не отправлена. проблема с соединением")

    def get(self, key):
        request = bytes(f"get {key}\n", encoding='utf-8')
        try:
            self.sock.sendall(request)
        except ClientError:
            print("не работает")

        while True:
            try:
                data = self.sock.recv(1024)
            except ClientError:
                raise ClientError

            if not data:
                continue

            if data.decode("utf-8") == "ok\n\n":
                return {}
                break
            elif data.decode("utf-8").startswith("error"):
                raise ClientError
            else:
                data = data.decode("utf-8")[3:-2:].split("\n")
                dict = {}
                for i in data:
                    temp = i.split(" ")
                    if temp[0] in dict.keys():
                        x = dict[temp[0]]
                        x.append((int(temp[2]), float(temp[1])))
                        dict[temp[0]] = x
                    else:
                        dict[temp[0]] = [(int(temp[2]), float(temp[1]))]

                return dict
                break


class ClientError(BaseException):

    def __init__(self):
        pass
