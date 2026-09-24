import torch
from .metrics import ClassificationMetrics


def create_optimizers(model, config):
    parameters = list(model.feature_extractor.parameters()) + list(model.target_classifier.parameters())
    if isinstance(model.mi_lambda, torch.nn.Parameter):
        parameters.append(model.mi_lambda)
    target = torch.optim.Adam(parameters, lr=config.lr)
    domain = torch.optim.Adam(model.domain_classifier.parameters(), lr=config.lr * config.domain_lr_factor)
    kwargs = dict(factor=config.scheduler_factor, patience=config.scheduler_patience,
                  threshold=config.scheduler_threshold, cooldown=config.scheduler_cooldown)
    schedulers = (torch.optim.lr_scheduler.ReduceLROnPlateau(target, mode='max', **kwargs),
                  torch.optim.lr_scheduler.ReduceLROnPlateau(domain, mode='min', **kwargs))
    return (target, domain), schedulers


def _check_loss(loss):
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite loss; training stopped')


def train_batch(model, x, labels, domains, optimizers, grad_clip_norm=1.0):

    target_opt, domain_opt = optimizers
    target_logits, domain_logits = model(x)
    target_loss = model.target_criterion(target_logits, labels)
    first_loss = model.first_step_loss(target_logits, domain_logits, labels)
    _check_loss(first_loss)
    target_opt.zero_grad(set_to_none=True)
    domain_opt.zero_grad(set_to_none=True)
    first_loss.backward()
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, error_if_nonfinite=True)
    target_opt.step()

    _, domain_logits = model(x, second_step=True)
    domain_loss = model.domain_loss(domain_logits, domains)
    _check_loss(domain_loss)
    target_opt.zero_grad(set_to_none=True)
    domain_opt.zero_grad(set_to_none=True)
    domain_loss.backward()
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, error_if_nonfinite=True)
    target_opt.step()
    domain_opt.step()
    model.apply_max_norm()
    return (target_logits.detach(), domain_logits.detach(),
            {'target_loss': target_loss.item(), 'first_step_loss': first_loss.item(),
             'domain_loss': domain_loss.item()})


def run_epoch(model, loader, device, *, optimizers=None, grad_clip_norm=1.0, target_only=False):
    training = optimizers is not None
    if training and target_only:
        raise ValueError('Training requires domain labels')
    model.train(training)
    target_metrics = ClassificationMetrics(model.target_classifier[1].out_features)
    domain_metrics = None if target_only else ClassificationMetrics(model.domain_classifier[1].out_features)
    totals, count = {}, 0
    with torch.set_grad_enabled(training):
        for x, batch_labels in loader:
            x = x.to(device)
            if isinstance(batch_labels, (tuple, list)):
                labels, domains = batch_labels
            else:
                labels, domains = batch_labels, None
            labels = labels.to(device)
            if not target_only:
                if domains is None or (domains < 0).any():
                    raise ValueError('Train/validation require known domain labels')
                domains = domains.to(device)
            if training:
                target, domain, losses = train_batch(model, x, labels, domains, optimizers, grad_clip_norm)
            else:
                target, domain = model(x)
                losses = {'target_loss': model.target_criterion(target, labels).item()}
                if not target_only:
                    losses.update(first_step_loss=model.first_step_loss(target, domain, labels).item(),
                                  domain_loss=model.domain_loss(domain, domains).item())
            target_metrics.update(target, labels)
            if domain_metrics is not None:
                domain_metrics.update(domain, domains)
            for name, value in losses.items():
                if not torch.isfinite(torch.tensor(value)):
                    raise FloatingPointError(f'Non-finite {name}')
                totals[name] = totals.get(name, 0.) + value * len(x)
            count += len(x)
    if not count:
        raise ValueError('Empty epoch; check dataset size, batch_size and drop_last')
    result = {name: value / count for name, value in totals.items()}
    result.update({'target_' + key: value for key, value in target_metrics.compute().items()})
    if domain_metrics is not None:
        result.update({'domain_' + key: value for key, value in domain_metrics.compute().items()})
    return result
