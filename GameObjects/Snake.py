import pygame
import random
import json
from typing import List, Tuple, Optional

# Colors (RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED   = (200, 0, 0)

# Directions
UP    = (0, -1)
DOWN  = (0, 1)
LEFT  = (-1, 0)
RIGHT = (1, 0)

class Snake:
    def __init__(self,
                 start_pos: Tuple[int, int],
                 grid_width: int,
                 grid_height: int):
        self.body: List[Tuple[int, int]] = [start_pos]
        self.direction: Tuple[int, int] = RIGHT
        self.grow_flag: bool = False
        self.grid_width = grid_width
        self.grid_height = grid_height

    def change_direction(self, new_direction: Tuple[int, int]) -> None:
        # Prevent reversing direction
        if (new_direction[0] * -1, new_direction[1] * -1) == self.direction:
            return
        self.direction = new_direction

    def move(self) -> None:
        head_x, head_y = self.body[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        self.body.insert(0, new_head)
        if not self.grow_flag:
            self.body.pop()
        else:
            self.grow_flag = False

    def grow(self) -> None:
        self.grow_flag = True

    def check_collision(self) -> bool:
        head = self.body[0]
        x, y = head
        # Wall collision
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return True
        # Self collision
        if head in self.body[1:]:
            return True
        return False


def generate_apple(snake_body: List[Tuple[int, int]],
                   grid_width: int,
                   grid_height: int) -> Tuple[int, int]:
    while True:
        pos = (random.randint(0, grid_width-1),
               random.randint(0, grid_height-1))
        if pos not in snake_body:
            return pos


def generate_apples(mode: int,
                    snake_body: List[Tuple[int, int]],
                    grid_width: int,
                    grid_height: int) -> List[Tuple[int, int]]:
    apples: List[Tuple[int, int]] = []
    if mode == 1:
        apples.append(generate_apple(snake_body, grid_width, grid_height))
    elif mode == 2:
        count = random.randint(2, 5)
        while len(apples) < count:
            pos = generate_apple(snake_body, grid_width, grid_height)
            if pos not in apples:
                apples.append(pos)
    return apples

class SnakeGame:
    def __init__(
        self,
        grid_width: int = 30,
        grid_height: int = 30,
        cell_size: int = 20,
        game_mode: int = 1
    ):
        self.grid_width   = grid_width
        self.grid_height  = grid_height
        self.cell_size    = cell_size
        self.game_mode    = game_mode
        self._init_game_state()

    def _init_game_state(self) -> None:
        start = (self.grid_width // 2, self.grid_height // 2)
        self.snake = Snake(start, self.grid_width, self.grid_height)
        self.apples = generate_apples(
            self.game_mode,
            self.snake.body,
            self.grid_width,
            self.grid_height
        )
        self.score = 0
        self.done = False

    def reset(self) -> None:
        self._init_game_state()

    def get_state(self) -> List[float]:
        head = self.snake.body[0]
        w, h = self.grid_width - 1, self.grid_height - 1
        dl = head[0] / w
        dr = (w - head[0]) / w
        du = head[1] / h
        dd = (h - head[1]) / h
        ax, ay = self.apples[0]
        dx = (ax - head[0]) / w
        dy = (ay - head[1]) / h
        return [dl, dr, du, dd, dx, dy]

    def step(self,
             action: int,
             render: bool = False) -> None:
        dirs = [UP, DOWN, LEFT, RIGHT]
        self.snake.change_direction(dirs[action])
        self.snake.move()

        # Collision check
        if self.snake.check_collision():
            self.done = True
            return

        # Apple consumption
        head = self.snake.body[0]
        if head in self.apples:
            self.snake.grow()
            self.score += 1
            self.apples.remove(head)
            if self.game_mode == 1 or (self.game_mode == 2 and not self.apples):
                self.apples = generate_apples(
                    self.game_mode,
                    self.snake.body,
                    self.grid_width,
                    self.grid_height
                )

        # Rendering
        if render:
            self._ensure_pygame()
            self._draw(self.screen)

    def _ensure_pygame(self) -> None:
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.grid_width * self.cell_size,
                 self.grid_height * self.cell_size)
            )
            pygame.display.set_caption('Snake Game')

    def _draw(self, screen) -> None:
        screen.fill(WHITE)
        # Grid lines
        for x in range(0, self.grid_width*self.cell_size, self.cell_size):
            pygame.draw.line(screen, BLACK, (x, 0), (x, self.grid_height*self.cell_size))
        for y in range(0, self.grid_height*self.cell_size, self.cell_size):
            pygame.draw.line(screen, BLACK, (0, y), (self.grid_width*self.cell_size, y))
        # Snake body
        for segment in self.snake.body:
            rect = pygame.Rect(
                segment[0]*self.cell_size,
                segment[1]*self.cell_size,
                self.cell_size, self.cell_size
            )
            pygame.draw.rect(screen, GREEN, rect)
        # Apples
        for apple in self.apples:
            rect = pygame.Rect(
                apple[0]*self.cell_size,
                apple[1]*self.cell_size,
                self.cell_size, self.cell_size
            )
            pygame.draw.rect(screen, RED, rect)
        # Score
        font = pygame.font.SysFont(None, 24)
        score_text = font.render(f"Score: {self.score}", True, BLACK)
        screen.blit(score_text, (5, 5))
        pygame.display.flip()

    def replay(self,
               states_path: str,
               delay: float = 0.1) -> None:
        with open(states_path, 'r') as f:
            states = json.load(f)

        pygame.init()
        screen = pygame.display.set_mode(
            (self.grid_width * self.cell_size,
             self.grid_height * self.cell_size)
        )
        pygame.display.set_caption('Snake Replay')
        clock = pygame.time.Clock()

        for state in states:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return
            self.snake.body = state['snake']
            self.apples = state['apples']
            self.score = state['score']
            self._draw(screen)
            clock.tick(1 / delay)

        pygame.quit()