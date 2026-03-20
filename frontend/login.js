// login.js - Handles both separate Login and Signup pages

document.addEventListener("DOMContentLoaded", () => {
    const loginForm = document.getElementById("login-form");
    const signupForm = document.getElementById("signup-form");

    // --- LOGIN LOGIC ---
    if (loginForm) {
        const loginErrorEl = document.getElementById("login-error");
        
        loginForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            loginErrorEl.hidden = true;
            
            const submitButton = loginForm.querySelector("button[type='submit']");
            const originalText = submitButton.innerHTML;
            submitButton.disabled = true;
            submitButton.innerHTML = "Signing in...";

            const formData = new URLSearchParams();
            formData.append("username", loginForm.email.value.trim());
            formData.append("password", loginForm.password.value);

            try {
                const response = await fetch("/login", {
                    method: "POST",
                    headers: { "Content-Type": "application/x-www-form-urlencoded" },
                    body: formData,
                    credentials: "include",
                });

                if (!response.ok) {
                    throw new Error(response.status === 403 ? "Invalid credentials." : "Login failed.");
                }

                window.location.href = "/chat";
            } catch (error) {
                loginErrorEl.textContent = error.message;
                loginErrorEl.hidden = false;
                submitButton.disabled = false;
                submitButton.innerHTML = originalText;
            }
        });
    }

    // --- SIGNUP LOGIC ---
    if (signupForm) {
        const signupErrorEl = document.getElementById("signup-error");
        const signupSuccessEl = document.getElementById("signup-success");

        signupForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            signupErrorEl.hidden = true;
            signupSuccessEl.hidden = true;

            const password = signupForm.password.value;
            const confirm = signupForm.passwordConfirm.value;

            if (password !== confirm) {
                signupErrorEl.textContent = "Passwords do not match.";
                signupErrorEl.hidden = false;
                return;
            }

            const submitButton = signupForm.querySelector("button[type='submit']");
            submitButton.disabled = true;

            try {
                const response = await fetch("/users/", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        email: signupForm.email.value.trim(),
                        password: password
                    }),
                });

                if (!response.ok) {
                    const msg = response.status === 409 ? "Email already exists." : "Signup failed.";
                    throw new Error(msg);
                }

                signupSuccessEl.hidden = false;
                setTimeout(() => { window.location.href = "/"; }, 2000);
            } catch (error) {
                signupErrorEl.textContent = error.message;
                signupErrorEl.hidden = false;
                submitButton.disabled = false;
            }
        });
    }
});
