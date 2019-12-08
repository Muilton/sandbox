import csv
import os
class CarBase:
    def __init__(self, car_type, brand, photo_file_name, carrying):
        self.car_type = car_type
        self.brand = brand
        self.photo_file_name = photo_file_name
        try:
            self.carrying = float(carrying)
        except ValueError:
            print(f"Неправильный формат данных 'carring' (float): {carring}")
        
    def get_photo_file_ext(self):
        return os.path.splitext(self.photo_file_name)[-1]


class Car(CarBase):
    def __init__(self, car_type, brand, passenger_seats_count, photo_file_name, body_whl, carrying, extra):
        super().__init__(car_type, brand, photo_file_name, carrying)
        try:
            self.passenger_seats_count = int(passenger_seats_count)
        except ValueError:
            print(f"Неправильный формат данных 'passenger_seats_count' (int): {passenger_seats_count}")
        


class Truck(CarBase):
    def __init__(self, car_type, brand, passenger_seats_count, photo_file_name, body_whl, carrying, extra):
        super().__init__(car_type, brand, photo_file_name, carrying)
        
        try:
            self.carrying = float(carrying)
        except ValueError:
            print(f"Неправильный формат данных 'carring' (float): {carring}")
            
        if body_whl != "":
            body_list = body_whl.split("x")
        else:
            body_list = ["0", "0", "0"]
        
        try:
            self.body_width = float(body_list[0])
            self.body_height = float(body_list[1])
            self.body_length = float(body_list[2])
        except ValueError:
            print(f"неправильный формат данных 'body_whl': {body_whl}")
    
    def get_body_volume(self):
        return self.body_width*self.body_height*self.body_length
        

class SpecMachine(CarBase):
    def __init__(self, car_type, brand, passenger_seats_count, photo_file_name, body_whl, carrying, extra):
        super().__init__(car_type, brand, photo_file_name, carrying)
        self.extra = extra
        

def get_car_list(csv_filename):
    with open(csv_filename, "r", encoding='utf-8') as csv_fd:
        reader = csv.reader(csv_fd, delimiter=';')
        next(reader)  # пропускаем заголовок
        car_list = []
        for car in reader:
            try:
                if car[0] and len(car) == 7:
                    car_list.append(car)
            except IndexError:
                print(f"Авто в список не добавлено! неправильный формат данных 'car': {car}")
        
    return make_car_list(car_list)
        
def make_car_list(car_list):
    result_car_list = []
    
    for car in car_list:
        if car[0] == "car":
            temp_car = Car(car[0], car[1], car[2], car[3], car[4], car[5], car[6])
            result_car_list.append(temp_car)
        
        elif car[0] == "truck":
            temp_car = Truck(car[0], car[1], car[2], car[3], car[4], car[5], car[6])
            result_car_list.append(temp_car)
        
        elif car[0] == "spec_machine":
            temp_car = SpecMachine(car[0], car[1], car[2], car[3], car[4], car[5], car[6])
            result_car_list.append(temp_car)
            
    return result_car_list