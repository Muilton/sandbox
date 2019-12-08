import socket
import threading

dict = {}


def run_server(host, port):
    with socket.socket() as sock:
        sock.bind((host, port))
        sock.listen()
        while True:
            conn, addr = sock.accept()
            th = threading.Thread(target=new_proccess, args=(conn, addr))
            th.start()


def new_proccess(conn, addr):
    with conn:
        while True:
            message = conn.recv(1024)
            if not message:
                break
            message = message.decode('utf-8')
            print(message)
            request = type_of_request(message)
            print("REQUEST: ", request)
            conn.sendall(request)


def type_of_request(message):
    type = message[:3]
    if type == "put":
        return put(message[4:-1])
    elif type == "get":
        return get(message[4:-1])
    else:
        return bytes("error\nwrong command\n\n", encoding='utf-8')


def put(data):
    data = data.split(" ")
    if data[0] in dict.keys():
        x = dict[data[0]]
        new_pair = (int(data[2]), float(data[1]))
        if new_pair not in x:
            x.append(new_pair)
        dict[data[0]] = x
    else:
        dict[data[0]] = [(int(data[2]), float(data[1]))]

    return bytes("ok\n\n", encoding='utf-8')


def get(data):
    result, suffix = "ok\n", "\n"
    if data == "*":
        for i in dict:
            for j in dict[i]:
                result = result + f"{i} {j[1]} {j[0]}\n"

    elif data in dict.keys():
        for j in dict[data]:
            result = result + f"{data} {j[1]} {j[0]}\n"
    return bytes(result + suffix, encoding='utf-8')

# run_server("127.0.0.1", 12121)
