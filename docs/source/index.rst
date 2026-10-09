.. _landing-page:

.. raw:: html

   <div class="hero">
      <h1>Graphix</h1>

      <div class="hero-tagline">
         An open-source Python library for building, optimizing, and simulating measurement-based quantum computations (MBQC). Develop your MBQC workflow with the help of our modular, extensible, and user-friendly software framework.
      </div>
   </div>

.. raw:: html

   <div class="section-divider"></div>

Why Graphix?
------------

Whether you are learning MBQC, developing protocols and algorithms, or testing new applications of measurement-based quantum systems, Graphix provides a comprehensive architecture to support your work.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc
      :class-card: index-card

      Install Graphix and customize your local environment.

   .. grid-item-card:: Getting started
      :link: getting_started
      :link-type: doc
      :class-card: index-card

      Learn the basics of Graphix and run your first MBQC program.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc
      :class-card: index-card

      Dive into the core MBQC objects implemented in Graphix, and learn to run various computations with guided examples.


   .. grid-item-card:: Research workflows
      :link: research_workflows/index
      :link-type: doc
      :class-card: index-card

      Discover examples of recent work in MBQC reproduced and extended by simulations in Graphix.


.. raw:: html

   <div class="section-divider"></div>

How to cite us
--------------

If you use Graphix in your research, please consider citing the `latest paper <https://arxiv.org/abs/2608.24781>`_ and the accompanying `software release <https://zenodo.org/records/21997813>`_.

**Papers**

   [1] M. Uldemolins, P. Nair, E. Graham, S. Sunami, T. Martinez, and M. Garnier, *Graphix: A software framework for Measurement-Based Quantum Computation*, arXiv:2608.24781 (2026).

   .. tab-set::

      .. tab-item:: BibTeX

         .. code-block:: bibtex

            @article{uldemolins2026_graphix,
               title={Graphix: A software framework for Measurement-Based Quantum Computation},
               author={Uldemolins, Mateo and Nair, Pranav and Graham, Emlyn and Sunami, Shinichi and Martinez, Thierry and Garnier, Maxime},
               journal={arXiv preprint arXiv:2608.24781},
               year={2026}
            }

   [2] S. Sunami, M. Fukushima. *Graphix: optimizing and simulating measurement-based quantum computation on local-Clifford decorated graph*, arxiv:2212.11975 (2022).

**Software**

   M. Uldemolins, M. Fukushima, E. Graham, P. Nair, D. Sasaki, S. Shiratani, 
   Y. Watanabe, T. Martinez, M. Garnier, and S. Sunami, "Graphix" (v0.4), 
   Zenodo (2026), `<https://doi.org/10.5281/zenodo.21997813>`_

   .. tab-set::

      .. tab-item:: BibTeX

         .. code-block:: bibtex

            @software{graphix_v04,
               author    = {Uldemolins, Mateo and Fukushima, Masato and
                           Graham, Emlyn and Nair, Pranav and Sasaki, Daichi and
                           Shiratani, Sora and Watanabe, Yuki and
                           Martinez, Thierry and Garnier, Maxime and Sunami, Shinichi},
               title     = {Graphix},
               version   = {0.4},
               year      = {2026},
               publisher = {Zenodo},
               doi       = {10.5281/zenodo.21997813},
            }


.. raw:: html

   <div class="section-divider"></div>

   <div class="hero-buttons">
      <a class="sd-btn sd-btn-primary" href="https://github.com/TeamGraphix/graphix">
         GitHub
      </a>
      <a class="sd-btn sd-btn-secondary" href="development/apiref/index.html">
         API reference
      </a>
      <a class="sd-btn sd-btn-secondary" href="plugins/index.html">
         Plugins
      </a>
   </div>


.. toctree::
   :hidden:
   :maxdepth: 2

   installation
   getting_started
   tutorials/index
   research_workflows/index
   development/contributing
   development/compatibility
   development/apiref/index
   plugins/index
