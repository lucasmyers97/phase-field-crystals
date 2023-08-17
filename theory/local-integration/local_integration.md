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

- At each locally-owned cell, we must traverse neighbors (and neighbors-of-neighbors, etc.) until `is_in_neighorhood` returns false.
    - We assume that the neighborhood is connected to the original cell.
    - Each cell in the neighborhood should be passed `calculate_local_quantity`, it should evaluate this function, and then it should return `local_quantity`.

- If the neighborhood runs up against a ghost cell, need to invoke the external process protocol

- If the neighborhood runs up against a periodic boundary, need to traverse neighbors of periodic neighbor cell.
Invoke external process protocol if any cell one runs into is a ghost cell.

### External process send protocol

- Check all bounding boxes and enumerate which mpi subdomains this particular cell intersects.
- For each mpi subdomain which the local subdomain needs to contact, create a package with all the relevant information (presumably `is_in_neighborhood` and `calculate_local_quantity` functions).
- Use `dealii::Utilities::MPI::ConsensusAlgorithms` to send the packages to the appropriate places.

### External process receive protocol

- Iterate through all boundary cells (iterate through all cells and just skip cells not on the boundary).
    - Go through all packages received and, if the current cell is in neighborhood of the package, carry out the local process on all locally-owned cells within the package neighborhood and take the package out of the set of packages (it will have been dealt with).
        - Note that cells from other subdomains will be covered because the packages will have intersected their bounding boxes as well.
    - If the current cell is not in the neighborhood of any of the packages, just continue.

- For each package, must keep track of who sent it, and then send back the `local_quantity` to the sender. 

### Consensus algorithm objects

- `RequestType`: this must contain all of the information that could be sent to another process.
    - More specifically, it should be a `std::map` whose keys are pairs of cell iterators and dof_indices, and whose values are pairs of `is_in_neighborhood` and `calculate_local_quantity`.
        - The reason for this is that each local integral must be associated with a particular dof on a particular cell so that the projection can be properly carried out.
    - One of these should be put together for each subdomain whose bounding_box has been intersected -- this should probably be done with a `std::map` so we can add subdomains freely.
    - `create_request` just queries the `std::map` for the `std::list`

- `AnswerType`: 

## Building implementation

### Testing on a single-MPI process domain

- 
