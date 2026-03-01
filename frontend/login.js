const loginForm = document.getElementById("login-form");
const loginErrorEl = document.getElementById("login-error");
const signupForm = document.getElementById("signup-form");
const signupErrorEl = document.getElementById("signup-error");
const signupSuccessEl = document.getElementById("signup-success");
const authTabs = document.querySelectorAll(".auth-toggle__tab");

function setActiveAuthTab(target) {
  authTabs.forEach((tab) => {
    const isActive = tab.dataset.target === target;
    tab.classList.toggle("auth-toggle__tab--active", isActive);
  });

  if (loginForm) {
    loginForm.classList.toggle("auth-form--visible", target === "login");
  }
  if (signupForm) {
    signupForm.classList.toggle("auth-form--visible", target === "signup");
  }
}

authTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.target;
    if (!target) return;
    setActiveAuthTab(target);
  });
});

if (loginForm) {
  loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    loginErrorEl.hidden = true;
    loginErrorEl.textContent = "";

    const submitButton = loginForm.querySelector("button[type='submit']");
    if (submitButton) {
      submitButton.disabled = true;
      submitButton.textContent = "Signing in…";
    }

    const email = loginForm.email.value.trim();
    const password = loginForm.password.value;

    const body = new URLSearchParams();
    body.append("username", email);
    body.append("password", password);

    try {
      const response = await fetch("/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body,
        credentials: "include",
      });

      if (!response.ok) {
        const message =
          response.status === 403
            ? "Invalid email or password."
            : "Unable to sign in. Please try again.";
        throw new Error(message);
      }

      window.location.href = "/chat";
    } catch (error) {
      loginErrorEl.hidden = false;
      loginErrorEl.textContent =
        error instanceof Error
          ? error.message
          : "Something went wrong. Please try again.";
    } finally {
      if (submitButton) {
        submitButton.disabled = false;
        submitButton.textContent = "Sign in";
      }
    }
  });
}

if (signupForm) {
  signupForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    signupErrorEl.hidden = true;
    signupErrorEl.textContent = "";
    signupSuccessEl.hidden = true;

    const submitButton = signupForm.querySelector("button[type='submit']");
    if (submitButton) {
      submitButton.disabled = true;
      submitButton.textContent = "Creating…";
    }

    const email = signupForm.email.value.trim();
    const password = signupForm.password.value;
    const passwordConfirm = signupForm.passwordConfirm.value;

    if (password !== passwordConfirm) {
      signupErrorEl.hidden = false;
      signupErrorEl.textContent = "Passwords do not match.";
      if (submitButton) {
        submitButton.disabled = false;
        submitButton.textContent = "Create account";
      }
      return;
    }

    try {
      const response = await fetch("/users/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        let message = "Unable to create account. Please try again.";
        if (response.status === 409) {
          message = "An account with that email already exists.";
        }
        throw new Error(message);
      }

      signupSuccessEl.hidden = false;
      signupForm.reset();
      setActiveAuthTab("login");
    } catch (error) {
      signupErrorEl.hidden = false;
      signupErrorEl.textContent =
        error instanceof Error
          ? error.message
          : "Something went wrong. Please try again.";
    } finally {
      if (submitButton) {
        submitButton.disabled = false;
        submitButton.textContent = "Create account";
      }
    }
  });
}

