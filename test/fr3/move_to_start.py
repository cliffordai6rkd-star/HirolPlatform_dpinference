from panda_py import Panda

a = Panda('192.168.1.206')
a.move_to_start()
state = a.get_state()
print("Current joint angles (rad):", state.q)

# b = Panda('192.168.1.206')
# b.move_to_start()
