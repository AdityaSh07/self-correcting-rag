from pwdlib import PasswordHash

password_hash = PasswordHash.recommended()
def hash(password: str):
    return password_hash.hash(password)

def verify(plain_password, hashed_password):
    if password_hash.verify(plain_password, hashed_password):
        return True
    else:
        return False