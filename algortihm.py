import math
import random
import sys
import os

import neat
import pygame

# Constants
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

CAR_WIDTH = 60    
CAR_HEIGHT = 60

CRASH_COLOR = (255, 255, 255, 255) # Color to indicate crash

generation_count = 0 # Generation counter

class Vehicle:

    def __init__(self):
        # Load and rotate vehicle sprite
        self.sprite = pygame.image.load('car.png').convert() # Convert speeds up rendering
        self.sprite = pygame.transform.scale(self.sprite, (CAR_WIDTH, CAR_HEIGHT))
        self.rotated_sprite = self.sprite 

        # Initial position and state
        self.position = [830, 920] 
        self.angle = 0
        self.speed = 0
        self.speed_initialized = False 

        self.center = [self.position[0] + CAR_WIDTH / 2, self.position[1] + CAR_HEIGHT / 2] # Calculate center

        self.sensors = [] 
        self.sensor_drawings = [] 

        self.alive = True 

        self.distance_traveled = 0 
        self.time_elapsed = 0 

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position) 
        self.draw_sensor_readings(screen) 

    def draw_sensor_readings(self, screen):
        for sensor in self.sensors:
            pos = sensor[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

    def check_collision(self, game_map):
        self.alive = True
        for corner in self.corners:
            if game_map.get_at((int(corner[0]), int(corner[1]))) == CRASH_COLOR:
                self.alive = False
                break

    def check_sensor(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        while not game_map.get_at((x, y)) == CRASH_COLOR and length < 300:
            length += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        distance = int(math.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2))
        self.sensors.append([(x, y), distance])
    
    def update(self, game_map):
        if not self.speed_initialized:
            self.speed = 20
            self.speed_initialized = True

        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], SCREEN_WIDTH - 120)

        self.distance_traveled += self.speed
        self.time_elapsed += 1
        
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], SCREEN_WIDTH - 120)

        self.center = [int(self.position[0]) + CAR_WIDTH / 2, int(self.position[1]) + CAR_HEIGHT / 2]

        half_length = 0.5 * CAR_WIDTH
        lt_corner = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * half_length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * half_length]
        rt_corner = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * half_length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * half_length]
        lb_corner = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * half_length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * half_length]
        rb_corner = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * half_length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * half_length]
        self.corners = [lt_corner, rt_corner, lb_corner, rb_corner]

        self.check_collision(game_map)
        self.sensors.clear()

        for degree in range(-90, 120, 45):
            self.check_sensor(degree, game_map)

    def get_sensor_data(self):
        sensor_data = [0, 0, 0, 0, 0]
        for i, sensor in enumerate(self.sensors):
            sensor_data[i] = int(sensor[1] / 30)
        return sensor_data

    def is_alive(self):
        return self.alive

    def get_reward(self):
        return self.distance_traveled / (CAR_WIDTH / 2)

    def rotate_center(self, image, angle):
        rect = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rect = rect.copy()
        rotated_rect.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rect).copy()
        return rotated_image


def run_simulation(genomes, config):
    nets = []
    vehicles = []

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0
        vehicles.append(Vehicle())

    clock = pygame.time.Clock()
    gen_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load('map.png').convert()

    global generation_count
    generation_count += 1

    frame_counter = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        for i, vehicle in enumerate(vehicles):
            output = nets[i].activate(vehicle.get_sensor_data())
            decision = output.index(max(output))
            if decision == 0:
                vehicle.angle += 10 
            elif decision == 1:
                vehicle.angle -= 10 
            elif decision == 2:
                if vehicle.speed - 2 >= 12:
                    vehicle.speed -= 2 
            else:
                vehicle.speed += 2 
        
        alive_count = 0
        for i, vehicle in enumerate(vehicles):
            if vehicle.is_alive():
                alive_count += 1
                vehicle.update(game_map)
                genomes[i][1].fitness += vehicle.get_reward()

        if alive_count == 0:
            break

        frame_counter += 1
        if frame_counter == 30 * 40: 
            break

        screen.blit(game_map, (0, 0))
        for vehicle in vehicles:
            if vehicle.is_alive():
                vehicle.draw(screen)
        
        gen_text = gen_font.render("Generation: " + str(generation_count), True, (0,0,0))
        gen_text_rect = gen_text.get_rect()
        gen_text_rect.center = (900, 450)
        screen.blit(gen_text, gen_text_rect)

        alive_text = alive_font.render("Still Alive: " + str(alive_count), True, (0, 0, 0))
        alive_text_rect = alive_text.get_rect()
        alive_text_rect.center = (900, 490)
        screen.blit(alive_text, alive_text_rect)

        pygame.display.flip()
        clock.tick(60) 

if __name__ == "__main__":
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats_reporter = neat.StatisticsReporter()
    population.add_reporter(stats_reporter)
    
    population.run(run_simulation, 1000)
