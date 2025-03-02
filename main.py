import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from bernoulli_solver import bernoulli_equation, secant_method, analytical_solution

st.set_page_config(page_title="Bernoulli Equation Solver", layout="wide")

def create_equation_section():
    st.header("Bernoulli's Equation Solver")
    st.markdown("""
    <style>
    .equation {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="equation">
    <h3>Bernoulli's Equation:</h3>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"P_1 + \frac{1}{2}\rho v_1^2 + \rho gh_1 = P_2 + \frac{1}{2}\rho v_2^2 + \rho gh_2")

    st.markdown("""
    We're solving for velocity at point 2 ($v_2$) given all other parameters.
    """)

def validate_inputs(P1, P2, rho, g, h1, h2, v1):
    """Validate input parameters and return explanation if invalid"""
    if rho <= 0:
        return False, "Density must be positive"
    if v1 < 0:
        return False, "Velocity must be non-negative"
    if P1 < 0 or P2 < 0:
        return False, "Pressures should typically be positive"
    return True, ""

def suggest_initial_guesses(v1, P1, P2, rho, g, h1, h2):
    """Suggest reasonable initial guesses based on physics"""
    try:
        v2_analytical = analytical_solution(P1, P2, rho, g, h1, h2, v1)
        return max(0.1, v2_analytical * 0.5), min(v2_analytical * 1.5, 20.0)
    except:
        # If analytical solution fails, use v1-based guesses
        return max(0.1, v1 * 0.5), min(v1 * 1.5, 20.0)

def create_input_section():
    with st.container():
        st.subheader("System Parameters")

        # Fluid properties
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Fluid Properties")
            rho = st.number_input(
                "Fluid Density (ρ) [kg/m³]",
                value=1000.0,
                min_value=0.1,
                help="Density of the fluid in kilograms per cubic meter [kg/m³]\nExample values:\n- Water: 1000 kg/m³\n- Air: 1.225 kg/m³",
                format="%.1f"
            )
        with col2:
            g = 9.81  # gravitational acceleration
            st.markdown("""
            **Gravitational Acceleration (g)**: 9.81 m/s²
            """)

        # Point 1 and 2 parameters
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Point 1 (Known Parameters)")
            P1 = st.number_input(
                "Pressure at Point 1 (P₁) [Pa]",
                value=101325.0,
                help="Pressure at point 1 in Pascals [Pa]\nExample values:\n- Atmospheric pressure: 101325 Pa\n- 1 bar = 100000 Pa",
                format="%.1f"
            )
            h1 = st.number_input(
                "Height at Point 1 (h₁) [m]",
                value=0.0,
                help="Height at point 1 in meters [m]\nMeasured from a reference level",
                format="%.2f"
            )
            v1 = st.number_input(
                "Velocity at Point 1 (v₁) [m/s]",
                value=2.0,
                min_value=0.0,
                help="Known velocity at point 1 in meters per second [m/s]\nTypical range: 0.1 - 20 m/s for water flow",
                format="%.2f"
            )

        with col2:
            st.markdown("#### Point 2 (Unknown Velocity)")
            P2 = st.number_input(
                "Pressure at Point 2 (P₂) [Pa]",
                value=101325.0,
                help="Pressure at point 2 in Pascals [Pa]\nExample values:\n- Atmospheric pressure: 101325 Pa\n- 1 bar = 100000 Pa",
                format="%.1f"
            )
            h2 = st.number_input(
                "Height at Point 2 (h₂) [m]",
                value=1.0,
                help="Height at point 2 in meters [m]\nMeasured from the same reference level as h₁",
                format="%.2f"
            )

    # Add suggested guesses
    suggested_x0, suggested_x1 = suggest_initial_guesses(v1, P1, P2, rho, g, h1, h2)

    # Numerical method parameters in an expander
    with st.expander("Numerical Method Parameters", expanded=False):
        st.markdown("""
        The secant method requires two initial guesses for v₂ [m/s]. Good guesses help the method converge faster.

        **Physics-based suggestions:**
        - From conservation of energy, v₂ should be similar to v₁
        - For a contracting pipe, v₂ > v₁
        - For an expanding pipe, v₂ < v₁

        **Recommended ranges:**
        - For subsonic flows: 0.1 - 20 m/s
        - Start with values around v₁
        - Keep guesses reasonably apart
        """)

        col1, col2 = st.columns(2)
        with col1:
            x0 = st.number_input(
                "Initial Guess 1 [m/s]",
                value=suggested_x0,
                min_value=0.1,
                help="First initial guess for v₂ in meters per second [m/s]",
                format="%.2f"
            )
        with col2:
            x1 = st.number_input(
                "Initial Guess 2 [m/s]",
                value=suggested_x1,
                min_value=0.1,
                help="Second initial guess for v₂ in meters per second [m/s]",
                format="%.2f"
            )

    return P1, P2, rho, g, h1, h2, v1, x0, x1

def plot_convergence(iterations):
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations_array = np.array(iterations)

    ax.semilogy([i[0] for i in iterations], 
                [abs(i[2]) for i in iterations], 
                'b.-', linewidth=2, markersize=8)

    ax.set_xlabel('Iteration Number')
    ax.set_ylabel('Absolute Error |f(x)|')
    ax.set_title('Convergence History')
    ax.grid(True, which="both", ls="-", alpha=0.2)

    # Add points and annotations
    for i, (iter_num, x_val, fx_val) in enumerate(iterations):
        ax.annotate(f'v₂ = {x_val:.2f}',
                   (iter_num, abs(fx_val)),
                   textcoords="offset points",
                   xytext=(0,10),
                   ha='center')

    return fig

def plot_flow_visualization(h1, h2, v1, v2):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot pipe
    pipe_width = 1.0
    ax.fill_between([0, 5], [h1-pipe_width/2, h2-pipe_width/2], 
                   [h1+pipe_width/2, h2+pipe_width/2], 
                   color='lightgray', alpha=0.3)

    # Plot flow arrows
    x1, x2 = 1, 4
    arrow_density = 20
    x_points = np.linspace(x1, x2, arrow_density)

    for x in x_points:
        y = h1 + (h2-h1)*(x-x1)/(x2-x1)
        dx = 0.2
        dy = 0.1

        # Scale arrow size with velocity
        v_local = v1 + (v2-v1)*(x-x1)/(x2-x1)
        scale = v_local / max(v1, v2)

        ax.arrow(x, y, dx*scale, dy*(h2-h1)/(x2-x1), 
                head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.6)

    # Add velocity labels
    ax.text(x1, h1+0.5, f'v₁ = {v1:.2f} m/s', ha='center')
    ax.text(x2, h2+0.5, f'v₂ = {v2:.2f} m/s', ha='center')

    # Add height labels
    ax.text(0.5, h1, f'h₁ = {h1:.2f} m', va='center')
    ax.text(4.5, h2, f'h₂ = {h2:.2f} m', va='center')

    ax.set_xlim(0, 5)
    ax.set_ylim(min(h1,h2)-1.5, max(h1,h2)+1.5)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Height (m)')
    ax.set_title('Flow Visualization')
    ax.grid(True, alpha=0.2)

    return fig

def main():
    create_equation_section()
    P1, P2, rho, g, h1, h2, v1, x0, x1 = create_input_section()

    # Solve button with custom styling
    solve_button = st.button("Solve for v₂", type="primary")

    if solve_button:
        # Validate inputs first
        valid, message = validate_inputs(P1, P2, rho, g, h1, h2, v1)
        if not valid:
            st.error(f"Invalid input parameters: {message}")
            return

        try:
            # Try analytical solution first
            v2_analytical = analytical_solution(P1, P2, rho, g, h1, h2, v1)

            # Calculate numerical solution
            params = (P1, P2, rho, g, h1, h2, v1)
            v2_numerical, iterations, converged = secant_method(bernoulli_equation, x0, x1, params)

            if converged and v2_numerical is not None:
                st.success(f"""
                ✨ Solutions found:
                - Analytical: v₂ = {v2_analytical:.3f} m/s
                - Numerical: v₂ = {v2_numerical:.3f} m/s
                - Relative difference: {abs(v2_analytical - v2_numerical)/v2_analytical*100:.6f}%
                """)

                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["Flow Visualization", "Convergence Plot", "Iteration Details"])

                with tab1:
                    st.pyplot(plot_flow_visualization(h1, h2, v1, v2_numerical))

                with tab2:
                    st.pyplot(plot_convergence(iterations))

                with tab3:
                    st.markdown("#### Iteration History")
                    iteration_data = []
                    for i, x, fx in iterations:
                        iteration_data.append({
                            "Iteration": i,
                            "v₂ Value (m/s)": f"{x:.6f}",
                            "Error": f"{abs(fx):.2e}"
                        })
                    st.table(iteration_data)
            else:
                st.error("""
                The numerical method did not converge. This might be because:
                1. The initial guesses are too far from the solution
                2. The problem might be physically unrealistic
                3. Numerical instabilities in the calculation

                Try:
                1. Using the suggested initial guesses
                2. Making sure your input parameters are physically reasonable
                3. If you know the approximate answer, use guesses around that value
                """)

                # Show analytical solution anyway if available
                st.info(f"Analytical solution suggests v₂ ≈ {v2_analytical:.3f} m/s")

        except ValueError as e:
            st.error(f"Error in calculation: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

    # Help section
    with st.expander("Help & Information"):
        st.markdown("""
        ### About the Solver
        This application solves for the unknown velocity (v₂) in Bernoulli's equation using both:
        - Analytical solution (direct formula)
        - Numerical solution (secant method)

        ### Assumptions
        - Steady flow
        - Incompressible fluid
        - Inviscid flow
        - Flow along a streamline

        ### Tips for Getting Good Results
        1. Ensure your input parameters are physically realistic
        2. For initial guesses, start with values around v₁
        3. If the solution doesn't converge, try:
           - Different initial guesses
           - Checking units (all inputs should be in SI units)
           - Verifying the physical setup makes sense
        """)

if __name__ == "__main__":
    main()
