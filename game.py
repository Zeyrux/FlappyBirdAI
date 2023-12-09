from random import randint

from model import DQN

import torch
import numpy as np
import pygame as pg

# General settings
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 800
FPS = 10
MAX_FRAMES = 250
BASE_REWARD = 1
REWARD_DEATH = -1

# Bird settings
BIRD_WIDTH = 80
BIRD_HEIGHT = 50
BIRD_DISTANCE_BORDER = 100
BIRD_FALL_ACCELERATION = 9.81
BIRD_SPEED_UP = -19

# Obstacle settings
OBSTACLE_WIDTH = 100
OBSTACLE_DISTANCE = 750
OBSTACLE_MAX_VERTICAL_DISTANCE = 500
OBSTACLE_SPEED = 200
OBSTACLE_GAP_SIZE = 200

# Colors
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 0, 0)


class Bird:
    def __init__(self, game: "Game") -> None:
        self.game = game
        self.y = SCREEN_HEIGHT / 2 - BIRD_HEIGHT / 2
        self.x = BIRD_DISTANCE_BORDER
        self.speed = 0

        self.pg_initiated = False
        self.image = None

    def update(self, jump: bool) -> None:
        if jump:
            self.speed += BIRD_SPEED_UP
        self.speed += BIRD_FALL_ACCELERATION
        self.y += self.speed

    def collision(self) -> bool:
        # check out of bounds
        if self.y > SCREEN_HEIGHT - BIRD_HEIGHT / 2:
            return True
        if self.y < -BIRD_HEIGHT / 2:
            return True
        # check obstacles
        for obstacle in self.game.obstacles:
            if (
                obstacle.x < self.x < obstacle.x + OBSTACLE_WIDTH
                or obstacle.x < self.x + BIRD_WIDTH < obstacle.x + OBSTACLE_WIDTH
            ):
                if not (
                    self.y > obstacle.height
                    and self.y + BIRD_HEIGHT < obstacle.height + OBSTACLE_GAP_SIZE
                ):
                    return True
        return False

    def init_pg(self):
        self.image = pg.image.load("assets/bird.jpg")
        self.image.set_colorkey(COLOR_WHITE)
        self.image = self.image.convert_alpha()
        self.image = pg.transform.scale(self.image, (BIRD_WIDTH, BIRD_HEIGHT))

    def draw(self):
        if not self.pg_initiated:
            self.init_pg()
            self.pg_initiated = True
        self.game.screen.blit(self.image, (self.x, self.y))


class Obstacle:
    def __init__(
        self,
        game: "Game",
        last_obstacle: "Obstacle",
    ) -> None:
        self.game = game
        self.created_new = False
        self.x = SCREEN_WIDTH
        if last_obstacle:
            if last_obstacle.height - OBSTACLE_MAX_VERTICAL_DISTANCE > 0:
                min_height = last_obstacle.height - OBSTACLE_MAX_VERTICAL_DISTANCE
            else:
                min_height = 0
            if (
                last_obstacle.height + OBSTACLE_MAX_VERTICAL_DISTANCE
                < SCREEN_HEIGHT - OBSTACLE_GAP_SIZE
            ):
                max_height = last_obstacle.height + OBSTACLE_MAX_VERTICAL_DISTANCE
            else:
                max_height = SCREEN_HEIGHT - OBSTACLE_GAP_SIZE
            self.height = randint(min_height, max_height)
        else:
            self.height = randint(0, SCREEN_HEIGHT - OBSTACLE_GAP_SIZE)

        self.pg_initiated = False

    def update(self):
        self.x -= OBSTACLE_SPEED / 100
        if self.x < -OBSTACLE_WIDTH:
            self.game.score += 1
            self.game.obstacles.remove(self)
        if self.created_new is False and self.x < SCREEN_WIDTH - OBSTACLE_DISTANCE:
            self.game.obstacles.append(Obstacle(self.game, self))
            self.created_new = True

    def init_pg(self):
        pass

    def draw(self):
        if not self.pg_initiated:
            self.init_pg()
            self.pg_initiated = True
        pg.draw.rect(
            self.game.screen,
            COLOR_WHITE,
            (self.x, 0, OBSTACLE_WIDTH, self.height),
        )
        pg.draw.rect(
            self.game.screen,
            COLOR_WHITE,
            (
                self.x,
                self.height + OBSTACLE_GAP_SIZE,
                OBSTACLE_WIDTH,
                SCREEN_HEIGHT - self.height - OBSTACLE_GAP_SIZE,
            ),
        )


class Game:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.score = 0
        self.obstacles = []
        self.bird = Bird(self)
        self.use_obstacles = None
        self.jump_next = False
        self.frames = 0

        self.pygame_initiated = False
        self.clock = self.screen = self.background = None
        self.start()

    def start(self, use_obstacles: bool = False, use_jump_next: bool = False):
        self.use_obstacles = use_obstacles
        self.use_jump_next = use_jump_next
        if use_obstacles:
            self.obstacles.append(Obstacle(self, None))

    def step(
        self, action: list[int, int] = None, use_agent: bool = True, draw: bool = False
    ):
        self.frames += 1
        if self.frames > MAX_FRAMES:
            return 0.5, True, self.score
        if action is not None:
            jump = bool(action[0])
        if not use_agent:
            if not self.pygame_initiated:
                self.restart_pygame()
                self.pygame_initiated = True
            jump = self.handle_events()
        self.update(jump)
        collision = self.bird.collision()
        reward = self.get_reward(collision)
        self.score = reward
        if not use_agent or draw:
            self.draw()
        return reward, collision, self.score

    def update(self, jump: bool):
        self.bird.update(jump)
        for obstacle in self.obstacles:
            obstacle.update()

    def get_state(self) -> np.ndarray:
        data_obstacles = []
        for obstacle in self.obstacles[:2]:
            data_obstacles.append(obstacle.x / SCREEN_WIDTH)
            data_obstacles.append(obstacle.height / (SCREEN_HEIGHT - OBSTACLE_GAP_SIZE))
        while len(data_obstacles) < 4:
            data_obstacles.append(-1)
            data_obstacles.append(-1)  # TODO: add  + data_obstacles
        return np.array([self.bird.y / SCREEN_HEIGHT, self.bird.speed / 100])

    def get_reward(self, collision: bool) -> float:
        reward = BASE_REWARD
        if self.use_obstacles:
            for obstacle in self.obstacles:
                if obstacle.x + OBSTACLE_WIDTH > self.bird.x:
                    break
            reward = (
                reward
                - abs(
                    (obstacle.height + OBSTACLE_GAP_SIZE / 2)
                    - (self.bird.y + BIRD_HEIGHT / 2)
                )
                / (SCREEN_HEIGHT / 2)
            ) ** 2
            if collision:
                reward += REWARD_DEATH
            return reward
        reward = (
            reward
            - abs(SCREEN_HEIGHT / 2 - (self.bird.y + BIRD_HEIGHT / 2))
            / (SCREEN_HEIGHT / 2)
        ) ** 2
        if collision:
            reward += REWARD_DEATH
        return reward

    def run(self, model=None):
        while True:
            if model:
                if not self.pygame_initiated:
                    self.restart_pygame()
                    self.pygame_initiated = True
                state = self.get_state()
                action = [0, 0]
                move = model(torch.Tensor(state))
                action[torch.argmax(move).item()] = 1
                _, done, _ = self.step(action, draw=True)
            else:
                action = None
                _, done, _ = self.step(action, use_agent=False)
            if done:
                break
            self.clock.tick(FPS)

    def restart_pygame(self):
        pg.init()
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.background = pg.image.load("assets/background.jpg")
        self.background = pg.transform.scale(
            self.background, (SCREEN_WIDTH, SCREEN_HEIGHT)
        )

    def handle_events(self):
        if self.use_jump_next:
            self.jump_next = not self.jump_next
            return self.jump_next
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.is_running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    return True
        return False

    def draw(self):
        self.screen.blit(self.background, (0, 0))
        for obstacle in self.obstacles:
            obstacle.draw()
        self.draw_score()
        self.bird.draw()
        pg.display.flip()

    def draw_score(self):
        font = pg.font.SysFont("Comic Sans MS", 30, True)
        text = font.render(f"{self.score:.2f}", True, COLOR_RED)
        self.screen.blit(text, (SCREEN_WIDTH / 2 - text.get_width() / 2, 50))


if __name__ == "__main__":
    run_model = True
    game = Game()
    if run_model:
        model = DQN(2, 8, 2)
        model.load_state_dict(torch.load("model/model.pth"))
        model.eval()
        game.start()
        game.run(model)
    else:
        game.start(use_jump_next=True)
        game.run()
    pg.quit()
