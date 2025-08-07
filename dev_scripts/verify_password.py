from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
print(pwd_context.verify("admin123", "$2b$12$GSZNpCpT4LMPWAmXqYFf5OIDujI1brieYtYZbJe6kp5pm67uc.EAC"))