const form = document.getElementById("login-form");
const errorEl = document.getElementById("login-error");

if (form) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorEl.hidden = true;
    errorEl.textContent = "";

    const submitButton = form.querySelector("button[type='submit']");
    if (submitButton) {
      submitButton.disabled = true;
      submitButton.textContent = "Signing in…";
    }

    const email = form.email.value.trim();
    const password = form.password.value;

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
        const message = response.status === 403 ? "Invalid email or password." : "Unable to sign in. Please try again.";
        throw new Error(message);
      }

      window.location.href = "/chat";
    } catch (error) {
      errorEl.hidden = false;
      errorEl.textContent =
        error instanceof Error ? error.message : "Something went wrong. Please try again.";
    } finally {
      if (submitButton) {
        submitButton.disabled = false;
        submitButton.textContent = "Sign in";
      }
    }
  });
}

