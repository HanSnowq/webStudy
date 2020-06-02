import pygame
import time


class Music:
    def init(self):
        pygame.mixer.init()

    def load(self, music_path):
        pygame.mixer.music.load(music_path)

    def open(self, loop, start):
        pygame.mixer.music.play(loop, start)

    def pause(self):
        pygame.mixer.music.pause()

    def unpause(self):
        pygame.mixer.music.unpause()

    def stop(self):
        pygame.mixer.music.stop()

    def volume(self, value):
        # value 取值范围是0.0--1.0
        pygame.mixer.music.set_volume(value)

    def is_playing(self):
        return pygame.mixer.music.get_busy()

    def which_command(self, num):
        music_path = 'E:/Data/Music/战争之后.mp3'
        self.init()
        is_play = self.is_playing()
        if not is_play and num == 5:  # 若未播放且指令为5，则加载音乐并开始播放
            self.load(music_path)
            self.open(0, 0.0)
        if is_play:  # 音乐在播放中
            if num == 1:  # 指令为1，则调低声音
                self.volume(0.2)
                print(pygame.mixer.music.get_volume())
            if num == 2:  # 指令为2，则调大声音
                self.volume(0.8)
                print(pygame.mixer.music.get_volume())
            if num == 3:  # 指令为3，则暂停
                self.pause()
            if num == 4:  # 指令为4，则继续播放
                self.unpause()
            if num == 0:  # 指令为0，则关闭音乐
                self.stop()

# music = Music()
# music.which_command(5)
# time.sleep(10)
# music.which_command(3)
# time.sleep(2)
# music.which_command(4)
# time.sleep(5)
# music.which_command(1)
# time.sleep(10)
# music.which_command(2)
# time.sleep(12)
# music.which_command(0)
