import numpy as np
import random
import pygame
import sys

# Parameters
GRID_SIZE = 20
TILE_COUNT = 20
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.1
EPISODES = 1000
MAX_STEPS = 200

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((GRID_SIZE * TILE_COUNT, GRID_SIZE * TILE_COUNT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)

# Snake Game Environment
class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(10, 10)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.food = self.random_food_position()
        self.score = 0
        self.game_over = False
        self.steps = 0

    def random_food_position(self):
        while True:
            food = (random.randint(0, TILE_COUNT - 1), random.randint(0, TILE_COUNT - 1))
            if food not in self.snake:
                return food

    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        dir_x, dir_y = self.direction

        state = [
            # Danger straight
            (dir_x == 1 and (head_x + 1 >= TILE_COUNT or (head_x + 1, head_y) in self.snake)) or
            (dir_x == -1 and (head_x - 1 < 0 or (head_x - 1, head_y) in self.snake)) or
            (dir_y == 1 and (head_y + 1 >= TILE_COUNT or (head_x, head_y + 1) in self.snake)) or
            (dir_y == -1 and (head_y - 1 < 0 or (head_x, head_y - 1) in self.snake)),

            # Danger right
            (dir_y == 1 and (head_x + 1 >= TILE_COUNT or (head_x + 1, head_y) in self.snake)) or
            (dir_y == -1 and (head_x - 1 < 0 or (head_x - 1, head_y) in self.snake)) or
            (dir_x == -1 and (head_y + 1 >= TILE_COUNT or (head_x, head_y + 1) in self.snake)) or
            (dir_x == 1 and (head_y - 1 < 0 or (head_x, head_y - 1) in self.snake)),

            # Danger left
            (dir_y == -1 and (head_x + 1 >= TILE_COUNT or (head_x + 1, head_y) in self.snake)) or
            (dir_y == 1 and (head_x - 1 < 0 or (head_x - 1, head_y) in self.snake)) or
            (dir_x == 1 and (head_y + 1 >= TILE_COUNT or (head_x, head_y + 1) in self.snake)) or
            (dir_x == -1 and (head_y - 1 < 0 or (head_x, head_y - 1) in self.snake)),

            # Move direction
            dir_x == 1, dir_x == -1, dir_y == 1, dir_y == -1,

            # Food location
            food_x > head_x, food_x < head_x, food_y > head_y, food_y < head_y
        ]
        return tuple(map(int, state))

    def step(self, action):
        if action == 0:  # Straight
            pass
        elif action == 1:  # Right turn
            self.direction = (-self.direction[1], self.direction[0])
        elif action == 2:  # Left turn
            self.direction = (self.direction[1], -self.direction[0])

        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])

        # Check collision
        if (new_head[0] < 0 or new_head[0] >= TILE_COUNT or
            new_head[1] < 0 or new_head[1] >= TILE_COUNT or
            new_head in self.snake):
            self.game_over = True
            reward = -10
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                self.score += 1
                self.food = self.random_food_position()
                reward = 10
            else:
                self.snake.pop()
                reward = -1

        self.steps += 1
        if self.steps > MAX_STEPS:
            self.game_over = True

        return self.get_state(), reward, self.game_over

    def render(self):
        screen.fill((50, 50, 50))  # Dark background

        # Draw grid lines
        for x in range(TILE_COUNT):
            pygame.draw.line(screen, (100, 100, 100), (x * GRID_SIZE, 0), (x * GRID_SIZE, GRID_SIZE * TILE_COUNT))
        for y in range(TILE_COUNT):
            pygame.draw.line(screen, (100, 100, 100), (0, y * GRID_SIZE), (GRID_SIZE * TILE_COUNT, y * GRID_SIZE))

        # Draw snake
        for i, segment in enumerate(self.snake):
            color = (0, 255, 0) if i == 0 else (0, 200, 0)  # Head is brighter green
            pygame.draw.rect(screen, color, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # Draw direction arrow on snake head
        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.direction
        arrow_length = GRID_SIZE // 3
        arrow_x = head_x * GRID_SIZE + GRID_SIZE // 2
        arrow_y = head_y * GRID_SIZE + GRID_SIZE // 2
        pygame.draw.line(screen, (255, 255, 255),
                         (arrow_x, arrow_y),
                         (arrow_x + dir_x * arrow_length, arrow_y + dir_y * arrow_length), 2)

        # Draw food
        pygame.draw.rect(screen, (255, 0, 0), (self.food[0] * GRID_SIZE, self.food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # Display score
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

        pygame.display.flip()

# Q-Learning Agent
class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.epsilon = EXPLORATION_RATE

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        q_values = [self.get_q_value(state, a) for a in range(3)]
        max_q = max(q_values)
        actions_with_max_q = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(actions_with_max_q)

    def learn(self, state, action, reward, next_state):
        old_q = self.get_q_value(state, action)
        next_max_q = max([self.get_q_value(next_state, a) for a in range(3)])
        new_q = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max_q - old_q)
        self.q_table[(state, action)] = new_q

# Training Loop with GUI
def train():
    env = SnakeGame()
    agent = QLearningAgent()

    for episode in range(EPISODES):
        env.reset()
        state = env.get_state()
        total_reward = 0

        while not env.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            # Render the game
            env.render()
            clock.tick(10)

        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Score: {env.score}")

    print("Training complete.")

# Run Training
if __name__ == "__main__":
    train()