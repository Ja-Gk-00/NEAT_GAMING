import pygame
import random
import sys

# Parametry gry
CELL_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 30
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE

# Kolory (RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (200, 0, 0)

# Kierunki ruchu
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class Snake:
    def __init__(self):
        # Początkowa pozycja w srodku planszy
        self.body = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = RIGHT
        self.grow_flag = False

    def change_direction(self, new_direction):
        # Zapobieganie cofnieciu sie
        if (new_direction[0] * -1, new_direction[1] * -1) == self.direction:
            return
        self.direction = new_direction

    def move(self):
        head_x, head_y = self.body[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        self.body.insert(0, new_head)
        if not self.grow_flag:
            self.body.pop()  # Usuwamy ogon, jesli nie rosniemy
        else:
            self.grow_flag = False

    def grow(self):
        # Ustawiamy flage, by w nastepnym ruchu nie usuwac ogona
        self.grow_flag = True

    def check_collision(self):
        head = self.body[0]
        # Kolizja ze scianami
        if head[0] < 0 or head[0] >= GRID_WIDTH or head[1] < 0 or head[1] >= GRID_HEIGHT:
            return True
        # Kolizja z wlasnym cialem
        if head in self.body[1:]:
            return True
        return False

def generate_apple(snake_body):
    """Generuje pojedyncze jablko w losowej pozycji, ktora nie koliduje z cialem weza."""
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in snake_body:
            return pos

def generate_apples(mode, snake_body):
    """Generuje jablka w zaleznosci od wybranego trybu:
       tryb 1: jedno jablko,
       tryb 2: losowa liczba jablek od 2 do 5.
    """
    apples = []
    if mode == 1:
        apples.append(generate_apple(snake_body))
    elif mode == 2:
        num_apples = random.randint(2, 5)
        for _ in range(num_apples):
            while True:
                pos = generate_apple(snake_body)
                if pos not in apples:
                    apples.append(pos)
                    break
    return apples

def draw_grid(screen):
    """Rysuje siatke na planszy."""
    for x in range(0, WINDOW_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, WINDOW_HEIGHT))
    for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (WINDOW_WIDTH, y))

def draw_snake(screen, snake):
    """Rysuje weza."""
    for segment in snake.body:
        rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, GREEN, rect)

def draw_apples(screen, apples):
    """Rysuje jablka."""
    for apple in apples:
        rect = pygame.Rect(apple[0] * CELL_SIZE, apple[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, RED, rect)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Snake - Tryby pojawiania sie jablek")
    clock = pygame.time.Clock()

    # Ustawienie trybu gry: 1 - pojedyncze jablko, 2 - kilka jablek.
    game_mode = 1

    snake = Snake()
    apples = generate_apples(game_mode, snake.body)
    score = 0

    font = pygame.font.SysFont(None, 24)

    running = True
    while running:
        clock.tick(5)  # Ustawienie predkosci gry (5 FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                # Sterowanie wezem
                if event.key == pygame.K_UP:
                    snake.change_direction(UP)
                elif event.key == pygame.K_DOWN:
                    snake.change_direction(DOWN)
                elif event.key == pygame.K_LEFT:
                    snake.change_direction(LEFT)
                elif event.key == pygame.K_RIGHT:
                    snake.change_direction(RIGHT)
                # Zmiana trybu gry podczas rozgrywki
                elif event.key == pygame.K_1:
                    game_mode = 1
                    apples = generate_apples(game_mode, snake.body)
                elif event.key == pygame.K_2:
                    game_mode = 2
                    apples = generate_apples(game_mode, snake.body)

        snake.move()

        # Sprawdzenie kolizji (ze scianą lub z samym sobą)
        if snake.check_collision():
            print("Game Over! Score:", score)
            running = False

        # Sprawdzenie, czy wąz zjadl jablko
        head = snake.body[0]
        if head in apples:
            snake.grow()
            score += 1
            apples.remove(head)
            # W trybie 1: po zjedzeniu jablka pojawia sie nowe;
            # w trybie 2: jesli wszystkie jablka zostaly zjedzone, generujemy nowe.
            if game_mode == 1 or (game_mode == 2 and len(apples) == 0):
                apples = generate_apples(game_mode, snake.body)

        # Rysowanie elementow gry
        screen.fill(WHITE)
        draw_grid(screen)
        draw_snake(screen, snake)
        draw_apples(screen, apples)

        # Wyswietlanie wyniku i aktualnego trybu gry
        score_text = font.render("Score: " + str(score), True, BLACK)
        mode_text = font.render("Tryb: " + ("1 - jedno jablko" if game_mode == 1 else "2 - kilka jablek"), True, BLACK)
        screen.blit(score_text, (5, 5))
        screen.blit(mode_text, (5, 25))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
