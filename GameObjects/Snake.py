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
    
    @property
    def head(self) -> Tuple[int, int]:
        return self.body[0]

    def direction_one_hot(self) -> List[float]:
        dirs = [UP, RIGHT, DOWN, LEFT]
        return [1.0 if self.direction==d else 0.0 for d in dirs]

    def change_direction(self, new_direction: Tuple[int, int]) -> None:
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

class SnakeGame:
    def __init__(
        self,
        grid_width: int = 10,
        grid_height: int = 10,
        cell_size: int = 20,
        game_mode: int = 1
    ):
        self.grid_width  = grid_width
        self.grid_height = grid_height
        self.cell_size   = cell_size
        self.game_mode   = game_mode
        self.reset()

    def reset(self) -> None:
        # Initialize snake at center
        start = (self.grid_width // 2, self.grid_height // 2)
        self.snake = Snake(start, self.grid_width, self.grid_height)
        self.apples = self._generate_apples()
        self.score = 0
        self.done = False

    def _generate_apple(self) -> Tuple[int, int]:
        # Place apple not on snake
        while True:
            pos = (random.randint(0, self.grid_width-1),
                   random.randint(0, self.grid_height-1))
            if pos not in self.snake.body:
                return pos

    def _generate_apples(self) -> List[Tuple[int, int]]:
        apples: List[Tuple[int, int]] = []
        if self.game_mode == 1:
            apples.append(self._generate_apple())
        else:
            count = random.randint(2, 5)
            while len(apples) < count:
                pos = self._generate_apple()
                if pos not in apples:
                    apples.append(pos)
        return apples

    def step(self, action: int, render: bool = False) -> None:
        # Determine new direction and move
        dirs = [UP, DOWN, LEFT, RIGHT]
        self.snake.change_direction(dirs[action])
        self.snake.move()

        # Wrap-around logic for wall-teleport mode (mode 2)
        head_x, head_y = self.snake.body[0]
        if self.game_mode == 2:
            head_x %= self.grid_width
            head_y %= self.grid_height
            self.snake.body[0] = (head_x, head_y)

        head = self.snake.body[0]
        if head in self.apples:
            self.snake.grow()
            self.score += 1
            self.apples.remove(head)
            if self.game_mode == 1 or (self.game_mode != 1 and not self.apples):
                self.apples = self._generate_apples()

        # Self-collision always ends game
        if head in self.snake.body[1:]:
            self.done = True
            return

        # Wall collision only in mode 1
        if self.game_mode == 1:
            if head_x < 0 or head_x >= self.grid_width or head_y < 0 or head_y >= self.grid_height:
                self.done = True
                return

        # Rendering if needed
        if render:
            self._ensure_pygame()
            self._draw()    

    def get_state(self) -> List[float]:
        grid = [0.0] * (self.grid_width * self.grid_height)
        for x, y in self.snake.body:
            idx = y * self.grid_width + x
            if 0 <= idx < len(grid):
                grid[idx] = -1.0
        for x, y in self.apples:
            idx = y * self.grid_width + x
            if 0 <= idx < len(grid):
                grid[idx] = 1.0
        return grid

    def _ensure_pygame(self) -> None:
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.grid_width * self.cell_size,
                 self.grid_height * self.cell_size)
            )
            pygame.display.set_caption('Snake Game')

    def _draw(self) -> None:
        self.screen.fill(WHITE)
        # Draw grid lines
        for x in range(0, self.grid_width*self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, BLACK, (x, 0), (x, self.grid_height*self.cell_size))
        for y in range(0, self.grid_height*self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, BLACK, (0, y), (self.grid_width*self.cell_size, y))
        # Draw snake
        for seg in self.snake.body:
            rect = pygame.Rect(seg[0]*self.cell_size, seg[1]*self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, GREEN, rect)
        # Draw apples
        for apple in self.apples:
            rect = pygame.Rect(apple[0]*self.cell_size, apple[1]*self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, RED, rect)
        # Draw score
        font = pygame.font.SysFont(None, 24)
        score_surf = font.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(score_surf, (5, 5))
        pygame.display.flip()

    def replay(self, states_path: str, delay: float = 0.1) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.grid_width * self.cell_size,
             self.grid_height * self.cell_size)
        )
        pygame.display.set_caption('Snake Replay')
        clock = pygame.time.Clock()

        # Load states
        with open(states_path, 'r') as f:
            states = json.load(f)

        # Play simulation
        for state in states:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            self.snake.body = state['snake']
            self.apples = state['apples']
            self.score = state['score']
            self._draw()
            clock.tick(1 / delay)

        # Cleanup
        pygame.quit()
        if hasattr(self, 'screen'):
            del self.screen