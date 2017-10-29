(function($) {

    network_type_select = $("select[name='network_type']");
    storage_results_row = $("div[class='form-row field-results_storage']");
    network_type_select.change(function() {
        select = $(this);

        if (select.val() == '1') {
            // Clustering Type
            storage_results_row.show();
        } else {
            storage_results_row.hide();
        }
    });
    network_type_select.change();

    node_type_selects = $("select[name$='node_type']");
    node_type_selects.change(function() {
        select = $(this);
        select_fieldset = select.closest("fieldset");
        stochastic_fieldset = select_fieldset.nextAll("fieldset:has(h2:contains('Stochastic Type'))"); 
        deterministic_fieldset = select_fieldset.nextAll("fieldset:has(h2:contains('Deterministic Type'))");
        visualization_fieldset = select_fieldset.nextAll("fieldset:has(h2:contains('Visualization'))");

        if (select.val() == '0') { 
            // Stochastic Type
            deterministic_fieldset.hide(); 
            stochastic_fieldset.show();
            visualization_fieldset.show();
        } else { 
            // Deterministic Type
            stochastic_fieldset.hide();
            deterministic_fieldset.show();
            visualization_fieldset.hide();
        }
    });
    node_type_selects.change();

    is_observable_cbs = $("input[name$='is_observable']");
    is_observable_cbs.change(function() {
        io_cb = $(this);
        io_fieldset = io_cb.closest("fieldset");
        columns_inline = io_fieldset.nextAll("div[id$='columns-group']"); 

        if (io_cb.is(':checked')) { 
            columns_inline.show();
        } else { 
            columns_inline.hide();
        }
    });
    is_observable_cbs.change();

    $(document).on('formset:added', function(event, $row, formsetName) {
        if (formsetName == 'nodes') {
            node_type_selects.change();
            is_observable_cbs.change();
        }
    });

    $(document).on('formset:removed', function(event, $row, formsetName) {
        // Row removed, nothing to do here yet
    });
    
})(django.jQuery);
