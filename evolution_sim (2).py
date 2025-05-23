import pygame
import numpy as np
import asyncio
import random
import platform

# Simulation parameters
NUM_SPECIES = 20
MIN_NODES = 5
MAX_NODES = 10
MIN_MUSCLES = 3
MAX_MUSCLES = 8
MIN_BONES = 1
MAX_BONES = 3
MIN_TENDONS = 1
MAX_TENDONS = 3
MIN_SPRING_LIGAMENTS = 1
MAX_SPRING_LIGAMENTS = 3
FRICTION_RANGE = [0.0, 0.9]
MUSCLE_LENGTH_RANGE = [30, 90]
BONE_LENGTH_RANGE = [10, 60]
TENDON_LENGTH_RANGE = [60, 120]
TENDON_SPRING_RANGE = [0.01, 0.1]
TENDON_MAX_FORCE = 0.5
SPRING_LIGAMENT_LENGTH_RANGE = [30, 90]
SPRING_LIGAMENT_SPRING_RANGE = [0.02, 0.2]
SPRING_LIGAMENT_MAX_FORCE = 0.7
CELL_WIDTH = 800
CELL_HEIGHT = 150
GENERATION_FRAMES = 1000
SURVIVOR_COUNT = 4
MOVEMENT_REWARD_SCALE = 0.1
GRAVITY = 0.2722
NODE_COLLISION_DISTANCE = 16
NODE_COLLISION_STRENGTH = 0.05
WEIGHT_RANGE = [0.1, 2.0]
MUSCLE_SMOOTHING_FACTOR = 0.1
MUSCLE_SWITCH_FRAMES = 10
FLASH_DURATION = 30
DEAD_FITNESS_THRESHOLD = 10
FPS = 60
MUTATION_RATE = 0.1
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CONTENT_HEIGHT = NUM_SPECIES * (CELL_HEIGHT + 37.5)  # 3750 pixels

# Utility functions
def random_gaussian():
    u = v = 0
    while u == 0: u = random.random()
    while v == 0: v = random.random()
    return np.sqrt(-2.0 * np.log(u)) * np.cos(2.0 * np.pi * v)

def random_choice(arr, k=1):
    copy = arr[:]
    result = []
    for _ in range(k):
        if not copy:
            break
        idx = random.randint(0, len(copy) - 1)
        result.append(copy.pop(idx))
    return result[0] if k == 1 else result

class Node:
    def __init__(self, x, y, friction, weight):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([0.0, 0.0])
        self.friction = friction
        self.weight = weight
        self.prev_pos = np.array([x, y], dtype=float)

    def update_movement(self):
        delta = np.linalg.norm(self.pos - self.prev_pos)
        self.prev_pos = self.pos.copy()
        return delta

class Muscle:
    def __init__(self, node1, node2, target_length, type_):
        self.node1 = node1
        self.node2 = node2
        self.target_length = target_length
        self.desired_target_length = target_length
        self.type = type_
        self.strength = 0.2 if type_ == "rigid" else 0.02
        self.last_switch_frame = 0

    def get_current_length(self):
        return np.linalg.norm(self.node2.pos - self.node1.pos)

class Bone:
    def __init__(self, node1, node2, length):
        self.node1 = node1
        self.node2 = node2
        self.length = length
        self.strength = 0.1

class Tendon:
    def __init__(self, node1, node2, max_length, spring_constant):
        self.node1 = node1
        self.node2 = node2
        self.max_length = max_length
        self.spring_constant = spring_constant
        self.max_force = TENDON_MAX_FORCE

    def get_current_length(self):
        return np.linalg.norm(self.node2.pos - self.node1.pos)

    def apply_force(self):
        dist_vec = self.node2.pos - self.node1.pos
        dist = np.linalg.norm(dist_vec)
        if dist > self.max_length and dist > 0:
            force_magnitude = min(self.spring_constant * (dist - self.max_length), self.max_force)
            force = (dist_vec / dist) * force_magnitude
            self.node1.vel += force
            self.node2.vel -= force

class SpringLigament:
    def __init__(self, node1, node2, rest_length, spring_constant):
        self.node1 = node1
        self.node2 = node2
        self.rest_length = rest_length
        self.spring_constant = spring_constant
        self.max_force = SPRING_LIGAMENT_MAX_FORCE

    def get_current_length(self):
        return np.linalg.norm(self.node2.pos - self.node1.pos)

    def apply_force(self):
        dist_vec = self.node2.pos - self.node1.pos
        dist = np.linalg.norm(dist_vec)
        if dist > self.rest_length and dist > 0:
            force_magnitude = min(self.spring_constant * (dist - self.rest_length), self.max_force)
            force = (dist_vec / dist) * force_magnitude
            self.node1.vel += force
            self.node2.vel -= force

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.array([[random_gaussian() * 0.5 for _ in range(hidden_size)] for _ in range(input_size)])
        self.weights2 = np.array([[random_gaussian() * 0.5 for _ in range(output_size)] for _ in range(hidden_size)])
        self.bias1 = np.array([random_gaussian() * 0.1 for _ in range(hidden_size)])
        self.bias2 = np.array([random_gaussian() * 0.1 for _ in range(output_size)])

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        hidden = self.sigmoid(np.dot(inputs, self.weights1) + self.bias1)
        output = self.sigmoid(np.dot(hidden, self.weights2) + self.bias2)
        return output

class Species:
    def __init__(self):
        self.nodes = []
        self.muscles = []
        self.bones = []
        self.tendons = []
        self.spring_ligaments = []
        self.nn = None
        self.initial_com_x = 0
        self.distance = 0
        self.total_movement = 0
        self.frame_count = 0
        self.flash_frames = 0
        self.generate()

    def generate(self):
        num_nodes = random.randint(MIN_NODES, MAX_NODES)
        for _ in range(num_nodes):
            x = random.uniform(50, 100)
            y = random.uniform(50, 100)
            friction = random.uniform(*FRICTION_RANGE)
            weight = random.uniform(*WEIGHT_RANGE)
            self.nodes.append(Node(x, y, friction, weight))
        num_muscles = random.randint(MIN_MUSCLES, MAX_MUSCLES)
        for _ in range(num_muscles):
            n1, n2 = random_choice(self.nodes, 2)
            target_length = random.uniform(*MUSCLE_LENGTH_RANGE)
            type_ = "rigid" if random.random() < 0.5 else "weak"
            self.muscles.append(Muscle(n1, n2, target_length, type_))
        num_bones = random.randint(MIN_BONES, MAX_BONES)
        for _ in range(num_bones):
            n1, n2 = random_choice(self.nodes, 2)
            length = random.uniform(*BONE_LENGTH_RANGE)
            self.bones.append(Bone(n1, n2, length))
        num_tendons = random.randint(MIN_TENDONS, MAX_TENDONS)
        for _ in range(num_tendons):
            n1, n2 = random_choice(self.nodes, 2)
            max_length = random.uniform(*TENDON_LENGTH_RANGE)
            spring_constant = random.uniform(*TENDON_SPRING_RANGE)
            self.tendons.append(Tendon(n1, n2, max_length, spring_constant))
        num_spring_ligaments = random.randint(MIN_SPRING_LIGAMENTS, MAX_SPRING_LIGAMENTS)
        for _ in range(num_spring_ligaments):
            n1, n2 = random_choice(self.nodes, 2)
            rest_length = random.uniform(*SPRING_LIGAMENT_LENGTH_RANGE)
            spring_constant = random.uniform(*SPRING_LIGAMENT_SPRING_RANGE)
            self.spring_ligaments.append(SpringLigament(n1, n2, rest_length, spring_constant))
        connected_nodes = set()
        for muscle in self.muscles:
            connected_nodes.add(muscle.node1)
            connected_nodes.add(muscle.node2)
        for bone in self.bones:
            connected_nodes.add(bone.node1)
            connected_nodes.add(bone.node2)
        for tendon in self.tendons:
            connected_nodes.add(tendon.node1)
            connected_nodes.add(tendon.node2)
        for ligament in self.spring_ligaments:
            connected_nodes.add(ligament.node1)
            connected_nodes.add(ligament.node2)
        self.nodes = [node for node in self.nodes if node in connected_nodes]
        if not self.nodes:
            x = random.uniform(50, 100)
            y = random.uniform(50, 100)
            friction = random.uniform(*FRICTION_RANGE)
            weight = random.uniform(*WEIGHT_RANGE)
            self.nodes.append(Node(x, y, friction, weight))
        self.nn = NeuralNetwork(len(self.nodes) * 2, 10, len(self.muscles))
        self.initial_com_x = self.compute_center_of_mass()[0]

    def compute_center_of_mass(self):
        if not self.nodes:
            return np.array([0.0, 0.0])
        com = np.mean([node.pos for node in self.nodes], axis=0)
        return com

    def update(self):
        self.frame_count += 1
        if self.flash_frames > 0:
            self.flash_frames -= 1
        inputs = []
        for node in self.nodes:
            inputs.extend([node.pos[0] / CELL_WIDTH, node.pos[1] / CELL_HEIGHT])
        outputs = self.nn.forward(np.array(inputs))

        for i, muscle in enumerate(self.muscles):
            current_length = muscle.get_current_length()
            if self.frame_count - muscle.last_switch_frame >= MUSCLE_SWITCH_FRAMES:
                muscle.desired_target_length = 90 if current_length < 60 else 30
                muscle.last_switch_frame = self.frame_count
            muscle.target_length += (muscle.desired_target_length - muscle.target_length) * MUSCLE_SMOOTHING_FACTOR
            dist_vec = muscle.node2.pos - muscle.node1.pos
            dist = np.linalg.norm(dist_vec)
            if dist > 0:
                force = (dist_vec / dist) * muscle.strength * (dist - muscle.target_length)
                muscle.node1.vel += force
                muscle.node2.vel -= force

        for bone in self.bones:
            dist_vec = bone.node2.pos - bone.node1.pos
            dist = np.linalg.norm(dist_vec)
            if dist > bone.length:
                force = (dist_vec / dist) * bone.strength * (dist - bone.length)
                bone.node1.vel += force
                bone.node2.vel -= force

        for tendon in self.tendons:
            tendon.apply_force()

        for ligament in self.spring_ligaments:
            ligament.apply_force()

        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                node1, node2 = self.nodes[i], self.nodes[j]
                dist_vec = node2.pos - node1.pos
                dist = np.linalg.norm(dist_vec)
                if dist < NODE_COLLISION_DISTANCE and dist > 0:
                    overlap = NODE_COLLISION_DISTANCE - dist
                    force = (dist_vec / dist) * NODE_COLLISION_STRENGTH * overlap
                    node1.vel -= force
                    node2.vel += force

        movement = 0
        for node in self.nodes:
            node.vel[1] += GRAVITY * node.weight
            node.vel *= 1 - node.friction
            node.vel *= 0.95
            node.pos += node.vel
            node.pos[0] = np.clip(node.pos[0], -1000, 1000)
            node.pos[1] = np.clip(node.pos[1], 0, CELL_HEIGHT)
            if node.pos[1] >= CELL_HEIGHT - 10:
                node.pos[1] = CELL_HEIGHT - 10
                node.vel[1] = min(node.vel[1], 0)
            movement += node.update_movement()
        self.total_movement += movement

        self.distance = self.compute_center_of_mass()[0] - self.initial_com_x

    def draw(self, surface, offset_x, offset_y, flash):
        com = self.compute_center_of_mass()
        translate_x = offset_x + CELL_WIDTH / 2 - com[0]
        translate_y = offset_y

        # Draw ground
        pygame.draw.line(surface, (0, 0, 0), (translate_x - 1000, translate_y + CELL_HEIGHT - 10),
                         (translate_x + 1000, translate_y + CELL_HEIGHT - 10), 1)

        # Draw spring-ligaments
        for ligament in self.spring_ligaments:
            pos1 = (int(ligament.node1.pos[0] + translate_x), int(ligament.node1.pos[1] + translate_y))
            pos2 = (int(ligament.node2.pos[0] + translate_x), int(ligament.node2.pos[1] + translate_y))
            pygame.draw.line(surface, (0, 255, 255), pos1, pos2, 3)

        # Draw tendons
        for tendon in self.tendons:
            pos1 = (int(tendon.node1.pos[0] + translate_x), int(tendon.node1.pos[1] + translate_y))
            pos2 = (int(tendon.node2.pos[0] + translate_x), int(tendon.node2.pos[1] + translate_y))
            pygame.draw.line(surface, (0, 0, 255), pos1, pos2, 3)

        # Draw bones
        for bone in self.bones:
            pos1 = (int(bone.node1.pos[0] + translate_x), int(bone.node1.pos[1] + translate_y))
            pos2 = (int(bone.node2.pos[0] + translate_x), int(bone.node2.pos[1] + translate_y))
            pygame.draw.line(surface, (0, 0, 0), pos1, pos2, 3)

        # Draw muscles
        for muscle in self.muscles:
            pos1 = (int(muscle.node1.pos[0] + translate_x), int(muscle.node1.pos[1] + translate_y))
            pos2 = (int(muscle.node2.pos[0] + translate_x), int(muscle.node2.pos[1] + translate_y))
            pygame.draw.line(surface, (255, 0, 0), pos1, pos2, 3)

        # Draw nodes
        for node in self.nodes:
            red = int(node.friction * 255)
            blue = int((node.weight - WEIGHT_RANGE[0]) / (WEIGHT_RANGE[1] - WEIGHT_RANGE[0]) * 255)
            color = (red, 255 - red, blue)
            pos = (int(node.pos[0] + translate_x), int(node.pos[1] + translate_y))
            pygame.draw.circle(surface, color, pos, 8)

        # Draw center of mass
        com_pos = (int(com[0] + translate_x), int(com[1] + translate_y))
        pygame.draw.circle(surface, (0, 255, 0), com_pos, 5)

        # Draw flash border
        if flash:
            pygame.draw.rect(surface, (0, 255, 0), (offset_x, offset_y, CELL_WIDTH, CELL_HEIGHT), 2)

    def get_fitness(self):
        return self.distance + MOVEMENT_REWARD_SCALE * self.total_movement

    def mutate(self):
        for node in self.nodes:
            if random.random() < MUTATION_RATE:
                node.pos += np.array([random_gaussian() * 10, random_gaussian() * 10])
            if random.random() < MUTATION_RATE:
                node.friction = np.clip(node.friction + random_gaussian() * 0.05, *FRICTION_RANGE)
            if random.random() < MUTATION_RATE:
                node.weight = np.clip(node.weight + random_gaussian() * 0.05, *WEIGHT_RANGE)
        for muscle in self.muscles:
            if random.random() < MUTATION_RATE:
                muscle.target_length = np.clip(muscle.target_length + random_gaussian() * 10, *MUSCLE_LENGTH_RANGE)
                muscle.desired_target_length = muscle.target_length
            if random.random() < MUTATION_RATE:
                muscle.type = "rigid" if random.random() < 0.5 else "weak"
                muscle.strength = 0.2 if muscle.type == "rigid" else 0.02
        for bone in self.bones:
            if random.random() < MUTATION_RATE:
                bone.length = np.clip(bone.length + random_gaussian() * 10, *BONE_LENGTH_RANGE)
        for tendon in self.tendons:
            if random.random() < MUTATION_RATE:
                tendon.max_length = np.clip(tendon.max_length + random_gaussian() * 10, *TENDON_LENGTH_RANGE)
            if random.random() < MUTATION_RATE:
                tendon.spring_constant = np.clip(tendon.spring_constant + random_gaussian() * 0.01, *TENDON_SPRING_RANGE)
        for ligament in self.spring_ligaments:
            if random.random() < MUTATION_RATE:
                ligament.rest_length = np.clip(ligament.rest_length + random_gaussian() * 10, *SPRING_LIGAMENT_LENGTH_RANGE)
            if random.random() < MUTATION_RATE:
                ligament.spring_constant = np.clip(ligament.spring_constant + random_gaussian() * 0.02, *SPRING_LIGAMENT_SPRING_RANGE)
        for i in range(self.nn.weights1.shape[0]):
            for j in range(self.nn.weights1.shape[1]):
                if random.random() < MUTATION_RATE:
                    self.nn.weights1[i][j] += random_gaussian() * 0.2
        for i in range(self.nn.weights2.shape[0]):
            for j in range(self.nn.weights2.shape[1]):
                if random.random() < MUTATION_RATE:
                    self.nn.weights2[i][j] += random_gaussian() * 0.2
        for i in range(len(self.nn.bias1)):
            if random.random() < MUTATION_RATE:
                self.nn.bias1[i] += random_gaussian() * 0.1
        for i in range(len(self.nn.bias2)):
            if random.random() < MUTATION_RATE:
                self.nn.bias2[i] += random_gaussian() * 0.1

def select_survivors(species_list):
    fitnesses = [max(s.get_fitness(), 0) for s in species_list]
    total = sum(fitnesses) + 1e-6
    probabilities = [f / total for f in fitnesses]
    survivors = []
    for _ in range(SURVIVOR_COUNT):
        r = random.random()
        sum_prob = 0
        for j, prob in enumerate(probabilities):
            sum_prob += prob
            if r <= sum_prob:
                survivors.append(species_list[j])
                break
    return survivors

async def main():
    global MUTATION_RATE
    try:
        pygame.init()
    except Exception as e:
        print(f"Pygame initialization failed: {e}")
        return

    try:
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Evolutionary Simulation")
    except Exception as e:
        print(f"Display setup failed: {e}")
        return

    clock = pygame.time.Clock()
    try:
        font = pygame.font.Font(None, 14)
    except Exception as e:
        print(f"Font loading failed: {e}")
        return

    species_list = [Species() for _ in range(NUM_SPECIES)]
    frame = 0
    is_paused = False
    target_fps = 60
    mutation_rate = MUTATION_RATE
    scroll_y = 0
    max_scroll = CONTENT_HEIGHT - WINDOW_HEIGHT  # 3150 pixels
    scrollbar_rect = pygame.Rect(WINDOW_WIDTH - 20, 50, 20, WINDOW_HEIGHT - 60)
    thumb_height = (WINDOW_HEIGHT / CONTENT_HEIGHT) * (WINDOW_HEIGHT - 60)
    dragging_thumb = False

    # UI setup
    pause_rect = pygame.Rect(10, 10, 100, 30)
    speed_rect = pygame.Rect(120, 10, 100, 20)
    mutation_rect = pygame.Rect(230, 10, 100, 20)
    speed_value = 60  # 30 to 120 FPS
    mutation_value = 0.1  # 0.05 to 0.5
    dragging_speed = False
    dragging_mutation = False

    running = True
    while running:
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if pause_rect.collidepoint(event.pos):
                    is_paused = not is_paused
                elif speed_rect.collidepoint(event.pos):
                    dragging_speed = True
                elif mutation_rect.collidepoint(event.pos):
                    dragging_mutation = True
                elif scrollbar_rect.collidepoint(event.pos):
                    thumb_y = scrollbar_rect.y + (scroll_y / max_scroll) * (scrollbar_rect.height - thumb_height)
                    thumb_rect = pygame.Rect(scrollbar_rect.x, thumb_y, scrollbar_rect.width, thumb_height)
                    if thumb_rect.collidepoint(event.pos):
                        dragging_thumb = True
                    else:
                        # Click above/below thumb to jump
                        if event.pos[1] < thumb_y:
                            scroll_y = max(0, scroll_y - WINDOW_HEIGHT)
                        else:
                            scroll_y = min(max_scroll, scroll_y + WINDOW_HEIGHT)
                elif event.button == 4:  # Mouse wheel up
                    scroll_y = max(0, scroll_y - 50)
                elif event.button == 5:  # Mouse wheel down
                    scroll_y = min(max_scroll, scroll_y + 50)
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging_speed = False
                dragging_mutation = False
                dragging_thumb = False
            elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                if dragging_speed:
                    x = max(speed_rect.left, min(event.pos[0], speed_rect.right))
                    speed_value = 30 + (x - speed_rect.left) / 100 * (120 - 30)
                if dragging_mutation:
                    x = max(mutation_rect.left, min(event.pos[0], mutation_rect.right))
                    mutation_value = 0.05 + (x - mutation_rect.left) / 100 * (0.5 - 0.05)
                if dragging_thumb:
                    dy = event.rel[1] * (max_scroll / (scrollbar_rect.height - thumb_height))
                    scroll_y = max(0, min(max_scroll, scroll_y + dy))

        screen.fill((255, 255, 255))

        if not is_paused:
            target_fps = int(speed_value)
            MUTATION_RATE = mutation_value

            for species in species_list:
                species.update()

            frame += 1
            if frame >= GENERATION_FRAMES:
                survivors = select_survivors(species_list)
                new_species_list = []
                for species in species_list:
                    if species.get_fitness() < DEAD_FITNESS_THRESHOLD:
                        parent = random_choice(survivors)
                        child = Species()
                        child.nodes = [Node(n.pos[0], n.pos[1], n.friction, n.weight) for n in parent.nodes]
                        node_map = {id(n): child.nodes[i] for i, n in enumerate(parent.nodes)}
                        child.muscles = [Muscle(node_map[id(m.node1)], node_map[id(m.node2)], m.target_length, m.type)
                                         for m in parent.muscles]
                        child.bones = [Bone(node_map[id(b.node1)], node_map[id(b.node2)], b.length)
                                       for b in parent.bones]
                        child.tendons = [Tendon(node_map[id(t.node1)], node_map[id(t.node2)], t.max_length, t.spring_constant)
                                         for t in parent.tendons]
                        child.spring_ligaments = [SpringLigament(node_map[id(l.node1)], node_map[id(l.node2)], l.rest_length, l.spring_constant)
                                                 for l in parent.spring_ligaments]
                        child.nn = NeuralNetwork(len(child.nodes) * 2, 10, len(child.muscles))
                        child.nn.weights1 = parent.nn.weights1.copy()
                        child.nn.weights2 = parent.nn.weights2.copy()
                        child.nn.bias1 = parent.nn.bias1.copy()
                        child.nn.bias2 = parent.nn.bias2.copy()
                        child.mutate()
                        child.initial_com_x = child.compute_center_of_mass()[0]
                        child.flash_frames = FLASH_DURATION
                        new_species_list.append(child)
                    else:
                        species.mutate()
                        species.initial_com_x = species.compute_center_of_mass()[0]
                        species.distance = 0
                        species.total_movement = 0
                        species.frame_count = 0
                        species.flash_frames = FLASH_DURATION
                        new_species_list.append(species)
                species_list = new_species_list
                frame = 0

        try:
            # Draw species
            for i, species in enumerate(species_list):
                offset_y = i * (CELL_HEIGHT + 37.5) - scroll_y
                # Only draw if cell is visible
                if offset_y + CELL_HEIGHT + 37.5 >= 0 and offset_y <= WINDOW_HEIGHT:
                    species.draw(screen, 0, offset_y, species.flash_frames > 0)

                    # Draw fitness meter
                    meter_text = f"Distance: {species.distance:.2f} px | Movement: {species.total_movement * MOVEMENT_REWARD_SCALE:.2f} | Fitness: {species.get_fitness():.2f}"
                    text_surface = font.render(meter_text, True, (0, 0, 0))
                    text_rect = text_surface.get_rect(center=(CELL_WIDTH / 2, offset_y + CELL_HEIGHT + 18.75))
                    if text_rect.top >= 0 and text_rect.bottom <= WINDOW_HEIGHT:
                        screen.blit(text_surface, text_rect)

            # Draw UI
            pygame.draw.rect(screen, (0, 0, 255), pause_rect)
            pause_text = font.render("Resume" if is_paused else "Pause", True, (255, 255, 255))
            screen.blit(pause_text, (pause_rect.x + 10, pause_rect.y + 5))
            pygame.draw.rect(screen, (200, 200, 200), speed_rect)
            speed_pos = speed_rect.left + (speed_value - 30) / (120 - 30) * 100
            pygame.draw.rect(screen, (0, 0, 255), (speed_pos, speed_rect.y, 10, 20))
            speed_text = font.render(f"Speed: {int(speed_value)} FPS", True, (0, 0, 0))
            screen.blit(speed_text, (speed_rect.x, speed_rect.y - 20))
            pygame.draw.rect(screen, (200, 200, 200), mutation_rect)
            mutation_pos = mutation_rect.left + (mutation_value - 0.05) / (0.5 - 0.05) * 100
            pygame.draw.rect(screen, (0, 0, 255), (mutation_pos, mutation_rect.y, 10, 20))
            mutation_text = font.render(f"Mutation: {mutation_value:.2f}", True, (0, 0, 0))
            screen.blit(mutation_text, (mutation_rect.x, mutation_rect.y - 20))

            # Draw scrollbar
            pygame.draw.rect(screen, (200, 200, 200), scrollbar_rect)
            thumb_y = scrollbar_rect.y + (scroll_y / max_scroll) * (scrollbar_rect.height - thumb_height)
            pygame.draw.rect(screen, (0, 0, 255), (scrollbar_rect.x, thumb_y, scrollbar_rect.width, thumb_height))

        except Exception as e:
            print(f"Rendering error: {e}")

        pygame.display.flip()
        clock.tick(target_fps)
        await asyncio.sleep(1.0 / FPS)

    pygame.quit()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())