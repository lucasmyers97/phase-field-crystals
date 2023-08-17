---
title: "Local integration algorithm"
date: 2023-08-15T10:40:00-05:00 
author: "Lucas Myers"
---

# Local integration algorithm

## Goal

- At each quadrature point in every base cell, one should be able to query finite element function values on target cells in a neighborhood in order to calculate some value.
- Several things should be left to the user:
    - A function which takes a target cell and returns whether it's close enough to the quadrature point (can define a neighborhood in different ways)
        - `is_in_neighorhood(cell) -> bool`
    - A function which takes a bounding box and returns whether it's close enough to the quadrature point
        - `intersects_bounding_box(bounding_box) -> bool`
    - Which finite element function is needed (this should just be a vector)
        - `dealii::Vector` (may need to change this)
    - A function which takes a target cell and calculates the relevant local quantity on the target cell (e.g. the integral contribution from the cell) with reference to each of the quadrature points on a base base cell.
        - `calculate_local_quantity(cell) -> local_quantity` (may need to template this)
    - A function which takes all the `local_quantity` contributions from each target cell and does something with them on the base cell.

- The specialization in our case is as follows:
    - `is_in_neighborhood` checks the intersection of the quadrilateral target cell with a circle centered at a quadrature point of some given radius.
    - `intersects_bounding_box` does essentially the same thing, but with a slightly-differently-structured input.
    - `calculate_local_quantity` calculates:
    $$
    \sum_{q'} X^{(q')} \exp\left( -\frac{\left( \mathbf{r}^{(q)} - \mathbf{r}^{(q')} \right)^2}{2 a_0^2} \right) (J \times W)^{(q')}
    $$ 
    for all quadrature points $q$ on the base cell.
    The sum over $q'$ should only include points which are within a certain distance from $q$.
    Probably want to return a vector whose length is the `q_points` on the base cell.

## Implementation

1. Prepare functions for Consensus Algorithm.
    - Find all cells on current subdomain which have at least one quadrature point whose neighborhood intersects other bounding boxes.
        - For this, keep `std::map<subdomain_id_type, std::map<cell, quad_points>>` structure around, where `quad_points = std::vector<dealii::Point<dim>>`.
            - This assigns to each external subdomain which this subdomain may bump into (i.e. cells have quadrature points whose neighborhoods intersect their bounding boxes) a `std::map<cell, quad_points>>`.
            - The `std::map<cell, quad_points>` will be sent over to other domains.
            It keeps track of all the cells which intersect a given subdomain (from a particular other subdomain), and holds the list of quadrature points on that cell.
        - If the cell has an edge which is a periodic boundary, find which subdomains its periodic neighbor intersects.
    
    - Generate `create_request` function which queries a `std::map<subdomain_id_type, std::map<cell, quad_points>>` type using the inputted subdomain id (which I guess is just an `unsigned int`).

    - Generate `answer_request`.
        - This function loops through all boundary cells, and loops through each element of `std::map<cell, quad_points>` to see whether it intersects with a quadrature point's neighborhood.
        - If it does, it calculates the local quantity on that cell and then also goes to its neighbor cells until it collects all local quantities which are within the base cell's neighborhood.
        - The answer that is generated is a `std::map<cell, local_quantities>` where `local_quantities = std::vector<local_quantity>` with each entry corresponding to each quadrature point.

    - Generate `process_answer`.
        - This loops through all cells in `std::map<cell, local_quantities>`, and calls the user-provided function which processes those local quantities.

2. Call whichever Consensus Algorithm (we have all the pieces, just do it).

3. Do operations on owned subdomain.
    - Loop through each base cell, and traverse all target cells intersected by a neighborhood of at least one quadrature cell.
    - On each target cell, calculate local_quantity and then process it according to user-provided function.

## Building implementation

1. Write initial `is_in_neighborhood` function.
    - This will just take in a cell, and check whether any of the quadrature points in the cell are within the appropriate distance of the reference quadrature point.
    - Test this with executable which takes in a point at the command-line, generates (coarse-ish) triangulation, sets up a finite element with dof-handler, then tests every single cell's quadrature points for inclusion in the neighborhood.
    - Output vtu with "is_in_neighborhood" value.
    - Output all quadrature points into csv.
    - Can check by looking in paraview at vtu and also plotting quadrature points and circle centered at inputted point.

2. Write initial `calculate_local_quantity` which just marks a cell as being in a particular subdomain.
    - Can test this by just applying the function to one particular cell.

3. Write function which goes to all neighbor cells, checking `is_in_neighborhood`, and then applying `calculate_local_quantity` if it is.
    - Need to figure out how to measure distance from a periodic cell.
    - Test by looking at vtu file again.

4. Write function which checks whether point is in range of any of the bounding boxes. 
    - Can check this by just creating a bounding box and seeing whether arbitrary points say their neighborhood intersects it.

5. Write program which does the integral on a non-distributed Triangulation.
    - This just loops through each quadrature point on each cell.
    - For each quadrature point, it calculates $\sum_{q'} X^{(q')} \exp \left( - \left( \mathbf{r}^{(q)} - \mathbf{(q')} \right)^2 / 2 a_0^2 \right) \left( J \times W \right)^{(q')}$ within a radius of $a_0$ (or some multiple of that, whatever).
    - It then takes that value to calculate $\left< \phi_i, \tilde{X} \right>$. 
    - We may solve that with the mass matrix and solve to see if we get something that gets smoothed out. 
    - Can try it with stress tensor to see whether we get something at defect points.

### Testing on a single-MPI process domain

- 
